# RFC：多说话人利用

* **作者**：刘展勋
* **状态**：Draft
* **日期**：2026.7.14

## 1. 目标

本方案希望实现：

* LLM知道谁在说话
* 结合轮次检测，对非感兴趣说话人不作回应

## 3. 方案

- 选型speaker diatrization模型
- 新增SpeakerDiatrization模型类型，或者为ASR模型添加带speaker标记的新实现
- 新增Speaker事件，或使ASRResultPartial和ASRResultFinal事件可选携带说话人信息
- LLM agent通过修改prompt理解多说话人

## 4. 风险

* 初步大体方案，工程细节需要研读代码后确定；在轮次检测做好后可制定针对多说话人的微调方案

## 5. 验收标准

* [ ] 系统可知道谁在说话