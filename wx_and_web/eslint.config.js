import uni from '@uni-helper/eslint-config'

export default uni(
  {
    unocss: true,
  },
  {
    files: ['**/*.js', '**/*.ts', '**/*.vue'],
    env: {
      browser: true,
      node: true,
    },
    rules: {
      'no-console': 'off',
    },
  }
)
