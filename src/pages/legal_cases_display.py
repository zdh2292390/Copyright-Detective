"""
Legal Cases Display Module

This module provides the UI for displaying landmark AI copyright cases
that illustrate the importance of copyright detection workflows.
"""

import streamlit as st


# Legal cases data
LEGAL_CASES = [
    {
        "title": "Bartz v. Anthropic PBC",
        "citation": "3:24-cv-05417, (N.D. Cal. Jun 23, 2025) ECF No. 231",
        "court": "N.D. California",
        "badge": "AI Training & Fair Use",
        "summary": "District court holds that Anthropic's use of books to train its Claude large language models and its use of purchased copies of books to create digital permanent library constitute fair use, but its use of pirated books to create such library does not constitute fair use.",
        "ruling": "Training LLMs on copyrighted works is transformative fair use; using pirated copies to build a library is not.",
        "link": "https://www.courtlistener.com/docket/69058235/bartz-v-anthropic-pbc/",
        "icon": "⚖️"
    },
    {
        "title": "Getty Images v. Stability AI",
        "citation": "[2025] EWHC 2863 (Ch), IL-2023-000007 (Nov 4, 2025)",
        "court": "UK High Court",
        "badge": "First UK AI Ruling",
        "summary": "First major UK ruling on copyright and trade mark laws applied to generative AI. The Court rejected the central copyright allegation, holding that AI model weights are not 'copies' of training images. However, limited trade mark infringement was found for watermark reproduction in earlier Stable Diffusion versions.",
        "ruling": "AI model weights are not infringing copies; watermark replication in outputs can constitute trade mark infringement.",
        "link": "https://www.judiciary.uk/judgments/getty-images-v-stability-ai/",
        "icon": "🖼️"
    },
    {
        "title": "The New York Times v. Microsoft & OpenAI",
        "citation": "1:23-cv-11195 (S.D.N.Y.)",
        "court": "S.D. New York",
        "badge": "Media vs AI",
        "summary": "District court allows direct and contributory copyright infringement and trademark dilution claims to proceed. News organizations alleged defendants train LLMs using copyrighted content, resulting in 'regurgitation' of large portions of their content. Court found plaintiffs sufficiently alleged material contribution and constructive knowledge of infringement.",
        "ruling": "Copyright and trademark claims proceed; DMCA and unfair competition claims dismissed as preempted.",
        "link": "https://www.courtlistener.com/docket/68117049/the-new-york-times-company-v-microsoft-corporation/",
        "icon": "📰"
    },
    {
        "title": "Authors Guild v. OpenAI Inc.",
        "citation": "1:23-cv-08292-SHS (S.D.N.Y. Sep 19, 2023)",
        "court": "S.D. New York",
        "badge": "Authors' Rights",
        "summary": "Class action by the Authors Guild and 17 prominent authors including George R.R. Martin, John Grisham, and Jodi Picoult. Plaintiffs allege OpenAI copied entire literary works without permission to train ChatGPT. Now centralized in MDL No. 3143 with discovery focusing on training datasets and model development.",
        "ruling": "Ongoing - Key test of fair use doctrine for AI training on copyrighted books.",
        "link": "https://www.courtlistener.com/docket/67810584/authors-guild-v-openai-inc/",
        "icon": "📚"
    },
    {
        "title": "Andersen v. Stability AI",
        "citation": "3:23-cv-00201 (N.D. Cal.)",
        "court": "N.D. California",
        "badge": "Artist Rights",
        "summary": "Class action by visual artists against Stability AI, DeviantArt, and Midjourney. Plaintiffs allege Stable Diffusion was trained on billions of copyrighted images without permission. Court allowed direct infringement claim against Stability to proceed but dismissed vicarious infringement and other claims with leave to amend.",
        "ruling": "Direct infringement claim proceeds; other claims dismissed with leave to amend for clarity on 'compressed copies' theory.",
        "link": "https://www.courtlistener.com/docket/66732129/andersen-v-stability-ai-ltd/",
        "icon": "🎨"
    },
    {
        "title": "Concord Music v. Anthropic",
        "citation": "3:23-cv-01092 (M.D. Tenn.)",
        "court": "M.D. Tennessee",
        "badge": "Music Lyrics",
        "summary": "Music publishers sued Anthropic claiming Claude AI reproduces copyrighted song lyrics. Court dismissed contributory and vicarious infringement claims, finding allegations that unidentified users 'might' prompt Claude for lyrics were insufficient. Preliminary injunction denied as overbroad.",
        "ruling": "Secondary infringement claims dismissed with leave to amend; injunction denied for lack of irreparable harm.",
        "link": "https://www.courtlistener.com/docket/67894459/concord-music-group-inc-v-anthropic-pbc/",
        "icon": "🎵"
    },
]


def render_legal_case_display_page():
    """Showcase real-world lawsuits that underscore memorization risk."""
    
    st.markdown('<h4 class="section-header">⚖️ Legal Cases Display</h4>', unsafe_allow_html=True)
    st.markdown(
        "Curated legal milestones that illustrate why Copyright Detective workflows are essential."
    )
    
    # Build the marquee HTML using string concatenation to avoid f-string issues
    cards_html_parts = []
    for case in LEGAL_CASES * 2:  # Duplicate for seamless loop
        icon = case["icon"]
        badge = case["badge"]
        title = case["title"]
        citation = case["citation"]
        summary = case["summary"]
        ruling = case["ruling"]
        court = case["court"]
        link = case["link"]
        
        card_html = (
            '<div class="legal-case-card">'
            f'<span class="legal-case-card__icon">{icon}</span>'
            '<div class="legal-case-card__badge">'
            '<span>⚡</span>'
            f'<span>{badge}</span>'
            '</div>'
            f'<div class="legal-case-card__title">{title}</div>'
            f'<div class="legal-case-card__citation">{citation}</div>'
            f'<div class="legal-case-card__summary">{summary}</div>'
            '<div class="legal-case-card__ruling">'
            '<span class="legal-case-card__ruling-icon">⚖️</span>'
            f'<span class="legal-case-card__ruling-text">{ruling}</span>'
            '</div>'
            '<div class="legal-case-card__footer">'
            '<div class="legal-case-card__court">'
            '<span>🏛️</span>'
            f'<span>{court}</span>'
            '</div>'
            f'<a href="{link}" target="_blank" rel="noopener noreferrer" class="legal-case-card__link">'
            'View Docket'
            '<span class="legal-case-card__link-arrow">→</span>'
            '</a>'
            '</div>'
            '</div>'
        )
        cards_html_parts.append(card_html)
    
    cards_html = "".join(cards_html_parts)
    
    marquee_html = (
        '<div class="legal-cases-marquee-container">'
        '<div class="legal-cases-section-header">'
        '<span class="legal-cases-section-header__icon">📋</span>'
        '<span class="legal-cases-section-header__title">Landmark AI Copyright Cases</span>'
        '</div>'
        '<div class="legal-cases-section-header__subtitle">'
        'Key lawsuits shaping the intersection of artificial intelligence and intellectual property law'
        '</div>'
        '<div class="legal-cases-marquee">'
        f'{cards_html}'
        '</div>'
        '<div class="legal-cases-pause-hint">'
        '💡 Hover over any card to pause scrolling and read details'
        '</div>'
        '</div>'
    )
    
    st.markdown(marquee_html, unsafe_allow_html=True)

