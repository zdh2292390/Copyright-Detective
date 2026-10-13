"""OAuth leaves embedded previews while application state stays on its origin."""
from __future__ import annotations

from html.parser import HTMLParser
import json
import re
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlparse

from src import auth


NODE = shutil.which("node")
LOGIN_URL = "https://auth.example/auth/v1/authorize?provider=github&redirect_to=https%3A%2F%2Fapp.example%2F"


class Controls(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


# The component and app are same-origin. The surrounding preview is cross-origin:
# its document/location cannot be read, but a user-initiated top navigation is allowed.
JS_FIXTURE = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const payload = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
function storage() {
    const values = new Map();
    return {
        getItem: key => values.has(key) ? values.get(key) : null,
        setItem: (key, value) => values.set(key, String(value)),
        removeItem: key => values.delete(key),
    };
}
function location(href) {
    let current = href;
    const replacements = [];
    return {
        get href() { return current; },
        set href(value) { current = String(value); },
        get search() { return new URL(current).search; },
        get hash() { return new URL(current).hash; },
        get origin() { return new URL(current).origin; },
        replace(value) { current = String(value); replacements.push(current); },
        replacements,
    };
}
const topNavigations = [];
const outer = {};
const outerLocation = {};
Object.defineProperty(outerLocation, 'href', {
    get() { throw new Error('SecurityError: cross-origin location'); },
    set(value) { topNavigations.push(String(value)); },
});
Object.defineProperty(outer, 'document', {
    get() { throw new Error('SecurityError: cross-origin document'); },
});
Object.defineProperty(outer, 'localStorage', {
    get() { throw new Error('SecurityError: cross-origin storage'); },
});
outer.location = outerLocation;
outer.parent = outer;
outer.top = outer;
const listeners = [];
const app = {
    location: location('https://app.example/?view=saved#input'),
    document: {addEventListener(...args) { listeners.push(args); }},
    localStorage: storage(),
    parent: outer,
    top: outer,
};
const component = {
    location: location('https://app.example/component'),
    document: {},
    localStorage: app.localStorage,
    parent: app,
    top: outer,
};
const context = vm.createContext({window: component, URL, URLSearchParams});
vm.runInContext(payload.script, context);
const bridge = component.__copyrightDetectiveAuth;
const window = component;
"""


class AuthNavigationTests(unittest.TestCase):
    def test_enabled_native_link_navigates_top_without_javascript(self):
        controls = Controls(auth._github_sign_in_link_html(LOGIN_URL))
        anchors = [attrs for tag, attrs in controls.elements if tag == "a"]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["href"], LOGIN_URL)
        self.assertEqual(anchors[0]["target"], "_top")
        self.assertFalse(any(name.startswith("on") for name in anchors[0]))
        self.assertNotIn("script", [tag for tag, _ in controls.elements])

    def test_disabled_control_cannot_navigate(self):
        controls = Controls(auth._github_sign_in_link_html(LOGIN_URL, disabled=True))
        self.assertFalse(any(tag == "a" for tag, _ in controls.elements))
        buttons = [attrs for tag, attrs in controls.elements if tag == "button"]
        self.assertEqual(len(buttons), 1)
        self.assertIn("disabled", buttons[0])
        self.assertEqual(buttons[0]["aria-disabled"], "true")
        self.assertNotIn("href", buttons[0])

    def test_login_url_cannot_inject_attributes_or_markup(self):
        url = LOGIN_URL + '&value=" onclick="alert(1)\"><script>bad</script>'
        controls = Controls(auth._github_sign_in_link_html(url))
        anchors = [attrs for tag, attrs in controls.elements if tag == "a"]
        self.assertEqual(anchors[0]["href"], url)
        self.assertNotIn("onclick", anchors[0])
        self.assertNotIn("script", [tag for tag, _ in controls.elements])

    def test_pkce_and_navigation_survive_native_link(self):
        verifier = "test-verifier_with-safe-characters"
        navigation = {
            "main_navigation": "Content Recall Detection",
            "content_recall_mode": "Document Memorization Detection",
            "api_key": "excluded-private-value",
        }
        callback = "https://app.example/?view=saved"
        url = "https://auth.example/auth/v1/authorize?" + urlencode({
            "provider": "github", "redirect_to": callback, "code_challenge": "existing-challenge",
        })
        url = auth._embed_verifier_in_oauth_url(url, verifier)
        with patch.object(auth, "st", SimpleNamespace(session_state=navigation)):
            url = auth._with_oauth_navigation(url)
        controls = Controls(auth._github_sign_in_link_html(url))
        href = next(attrs["href"] for tag, attrs in controls.elements if tag == "a")
        params = parse_qs(urlparse(href).query)
        self.assertEqual(params["code_challenge"], ["existing-challenge"])
        redirect = parse_qs(urlparse(params["redirect_to"][0]).query)
        self.assertEqual(redirect[auth.OAUTH_PKCE_PARAM], [verifier])
        self.assertEqual(redirect["view"], ["saved"])
        self.assertEqual(json.loads(redirect[auth.OAUTH_NAVIGATION_PARAM][0]), {
            key: value for key, value in navigation.items() if key != "api_key"
        })

    def run_bridge(self, checks, *, script=None):
        if not NODE:
            self.skipTest("Node is required for the browser bridge behavioral checks")
        result = subprocess.run(
            [NODE, "-e", JS_FIXTURE + checks],
            input=json.dumps({"script": script or auth._AUTH_CLIENT_SCRIPT}),
            text=True, capture_output=True, check=False, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_embedded_bridge_uses_readable_application_window_and_storage(self):
        self.run_bridge(r"""
assert.equal(bridge.appWindow(window), app);
assert.equal(bridge.appStorage(), app.localStorage);
bridge.persistPkceVerifier('saved-verifier');
assert.equal(bridge.readPkceVerifier(), 'saved-verifier');
bridge.persistAuthCache('access', 'refresh', 'owner');
assert.deepEqual(JSON.parse(app.localStorage.getItem('copyright_detective_auth')), {
    access_token: 'access', refresh_token: 'refresh', user_id: 'owner',
});
assert.equal(topNavigations.length, 0);
""")

    def test_standalone_application_is_still_the_bridge_target(self):
        self.run_bridge(r"""
app.parent = app;
app.top = app;
component.top = app;
assert.equal(bridge.appWindow(window), app);
assert.equal(bridge.appWindow(app), app);
assert.equal(bridge.appStorage(), app.localStorage);
""")

    def test_return_url_click_listener_is_bound_to_application_document(self):
        captured = []
        with patch.object(auth, "_auth_iframe", side_effect=lambda html, **kwargs: captured.append(html)):
            auth._bind_return_url_on_sign_in_click(LOGIN_URL)
        script = re.search(r"<script>(.*?)</script>", captured[0], re.DOTALL).group(1)
        self.run_bridge(r"""
assert.equal(listeners.length, 1);
assert.equal(listeners[0][0], 'click');
const link = {getAttribute: name => name === 'href' ? app.__copyrightDetectiveGitHubLoginUrl : null};
listeners[0][1]({target: {closest: () => link}});
assert.equal(app.localStorage.getItem('copyright_detective_return_url'), 'https://app.example/?view=saved');
assert.equal(topNavigations.length, 0);
""", script=script)

    def test_callback_restores_application_route_and_navigation(self):
        self.run_bridge(r"""
const navigation = JSON.stringify({main_navigation: 'Content Recall Detection'});
bridge.persistReturnUrl('https://app.example/?view=saved#input');
bridge.persistAuthCache('access', 'refresh', 'owner');
app.location.href = 'https://app.example/?code=callback&cd_cv=verifier';
assert.equal(bridge.restoreReturnUrlAfterLogin(window, navigation), true);
const restored = new URL(app.location.href);
assert.equal(restored.origin, 'https://app.example');
assert.equal(restored.searchParams.get('view'), 'saved');
assert.equal(restored.searchParams.get('cd_nav'), navigation);
assert.equal(restored.searchParams.has('code'), false);
assert.equal(restored.searchParams.has('cd_cv'), false);
assert.equal(app.localStorage.getItem('copyright_detective_return_url'), null);
assert.equal(bridge.readCachedAuth().access_token, 'access');
assert.equal(topNavigations.length, 0);
""")

    def test_javascript_login_stores_app_route_and_navigates_top(self):
        self.run_bridge(r"""
const oauthUrl = 'https://auth.example/auth/v1/authorize?provider=github';
bridge.signInWithGitHub(window, oauthUrl);
assert.equal(app.localStorage.getItem('copyright_detective_return_url'), 'https://app.example/?view=saved');
assert.deepEqual(topNavigations, [oauthUrl]);
assert.equal(app.location.href, 'https://app.example/?view=saved#input');
""")


if __name__ == "__main__":
    unittest.main()
