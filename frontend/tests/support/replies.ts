/**
 * Replies produced by the mock agent (see `tests/support/mock_models/mock_agent.py`).
 */

export const GREETING_REPLY = "你好，我是测试助手。";
export const STORY = ["一", "二", "三", "四", "五", "六", "七", "八"].map((word) => `故事第${word}句。`).join("");

/** Build the reply to `你刚才说了什么`. */
export function summaryReply(lastAssistantText: string | null, historyCount: number): string {
    const quoted = lastAssistantText ? lastAssistantText.replace(/。+$/, "") : "无";
    return `上一句是：${quoted}。历史共${historyCount}条。`;
}
