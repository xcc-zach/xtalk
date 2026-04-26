// commitlint.config.mjs
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'perf', 'refactor', 'chore'], // 包含所有自定义类型
    ],
  },
}
