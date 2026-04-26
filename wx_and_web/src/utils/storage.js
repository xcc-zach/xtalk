/**
 * 本地存储工具类
 */

const STORAGE_KEY = 'tetris_high_score'

/**
 * 获取最高分
 */
export function getHighScore() {
  try {
    const score = uni.getStorageSync(STORAGE_KEY)
    return score || 0
  }
  catch (e) {
    console.error('获取最高分失败:', e)
    return 0
  }
}

/**
 * 保存最高分
 */
export function saveHighScore(score) {
  try {
    const currentHighScore = getHighScore()
    if (score > currentHighScore) {
      uni.setStorageSync(STORAGE_KEY, score)
      return true
    }
    return false
  }
  catch (e) {
    console.error('保存最高分失败:', e)
    return false
  }
}

/**
 * 清除最高分
 */
export function clearHighScore() {
  try {
    uni.removeStorageSync(STORAGE_KEY)
    return true
  }
  catch (e) {
    console.error('清除最高分失败:', e)
    return false
  }
}
