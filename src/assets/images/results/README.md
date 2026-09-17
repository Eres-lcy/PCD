# 替换结果图

标题、图片路径和图注直接在 Vue 文件中修改：

- 仿真、真机结果：`src/components/Results.vue`。
- OOD 结果：`src/components/OodDemo.vue`，位于视频上方。

把截图放进本目录，修改对应 `<img>` 的 `src`；说明文字在 `<figcaption>` 中。
图片等比显示，不提供点击放大。
