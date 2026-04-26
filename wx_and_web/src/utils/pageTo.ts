/**
 * @description 页面跳转（非 tabBar 页面）
 */
export function jumpPageTo(
  options: { url: string, query?: Record<string, any> } | string,
): void {
  let finalUrl = ''
  let query: Record<string, any> | undefined

  if (typeof options === 'string') {
    finalUrl = options
  }
  else {
    finalUrl = options.url
    query = options.query
  }

  if (query) {
    const queryString = Object.entries(query)
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join('&')
    finalUrl += `?${queryString}`
  }

  uni.navigateTo({ url: finalUrl }).then(() => {})
}

/**
 * @description 页面跳转（tabBar 页面）
 */
export function jumpSwitchPageTo(
  options: { url: string, query?: Record<string, any> } | string,
): void {
  let finalUrl = ''
  let query: Record<string, any> | undefined

  if (typeof options === 'string') {
    finalUrl = options
  }
  else {
    finalUrl = options.url
    query = options.query
  }

  if (query) {
    const queryString = Object.entries(query)
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join('&')
    finalUrl += `?${queryString}`
  }

  uni.switchTab({ url: finalUrl })
}

/**
 * @description 页面返回
 */
export function jumpPageBack(options: { delta?: number } = { delta: 1 }): void {
  uni.navigateBack({
    delta: options.delta ?? 1,
  }).then(() => {})
}
