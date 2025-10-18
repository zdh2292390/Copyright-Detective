# Prompt Preview UI 更新说明

## 更改内容

将 `render_prompt_preview()` 函数从自定义 HTML/iframe 实现改为使用 Streamlit 官方原生组件。

## 修改的文件

- `src/components.py` - 重写了 `render_prompt_preview()` 函数

## 新实现特点

### 使用的官方组件
- **`st.expander()`** - 可折叠容器，替代原来的自定义 HTML details 元素
- **`st.code()`** - 代码显示块，内置复制按钮功能
- **`st.caption()`** - 显示统计信息（字符数和单词数）

### 优势
1. ✅ **更简洁** - 代码从 ~140 行减少到 ~13 行
2. ✅ **原生体验** - 使用 Streamlit 默认样式，与应用其他部分更一致
3. ✅ **内置功能** - `st.code()` 自带复制按钮，无需自定义 JavaScript
4. ✅ **更好维护** - 不依赖自定义 HTML/CSS/JS，减少潜在兼容性问题
5. ✅ **响应式** - 自动适配移动端和不同屏幕尺寸
6. ✅ **无障碍性** - Streamlit 组件自带更好的可访问性支持

### 功能保持
- ✅ 可折叠/展开（通过 `expanded` 参数）
- ✅ 自定义标题（通过 `title` 参数）
- ✅ 复制功能（`st.code()` 内置）
- ✅ 显示字符数和单词数统计

## 使用示例

```python
# 基本用法
render_prompt_preview("这是一个示例 prompt")

# 自定义标题
render_prompt_preview(
    "这是 prompt 内容",
    title="Custom Prompt Template"
)

# 默认展开
render_prompt_preview(
    "这是 prompt 内容",
    expanded=True
)
```

## 向后兼容性

✅ 完全向后兼容 - 所有现有调用点无需修改，函数签名保持不变。

## 测试建议

运行应用并测试以下页面的 prompt preview 功能：
1. **Direct Recall Test** → Text Snippet Analysis
   - Next-Passage Prediction
   - Prior-Context Reconstruction
   - Title Prediction
2. **Direct Recall Test** → Whole PDF Analysis
3. **Unlearning Detection Test** → Prompt-Based Probes
4. **Persuasive Jailbreak Test** → Evaluation Experiments

## 已知问题

无。新实现更稳定可靠。
