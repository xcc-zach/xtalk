import type { Rule } from 'unocss'
import { presetUni } from '@uni-helper/unocss-preset-uni'
import {
  defineConfig,
  presetIcons,
  transformerDirectives,
  transformerVariantGroup,
} from 'unocss'

function createSizeRules(): Rule[] {
  return [
    ['size-full', { width: '100%', height: '100%' }],
    ['size-screen', { width: '100vw', height: '100vh' }],
    [/^size-(\d+)$/, match => ({ width: `${match[1]}px`, height: `${match[1]}px` })],
    [/^size-\[(\d+)\]/, match => ({ width: `${match[1]}px`, height: `${match[1]}px` })],
    [/^size-(\d+)x(\d+)$/, match => ({ width: `${match[1]}px`, height: `${match[2]}px` })],
    [
      /^size-\[(\d+)([a-z]+)\]x\[(\d+)([a-z]+)\]/,
      match => ({
        width: `${match[1]}${match[2]}`,
        height: `${match[3]}${match[4]}`,
      }),
    ],
  ]
}

function createFlexRules(): Rule[] {
  return [
    [
      'flex-center',
      {
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
      },
    ],
    ['flex-x-center', { 'display': 'flex', 'justify-content': 'center' }],
    ['flex-y-center', { 'display': 'flex', 'align-items': 'center' }],
    ['flex-x-end', { 'display': 'flex', 'justify-content': 'flex-end' }],
    ['flex-y-end', { 'display': 'flex', 'align-items': 'flex-end' }],
  ]
}

function createPositionRules(): Rule[] {
  return [
    [
      'position-center',
      {
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: 'translate(-50%, -50%)',
      },
    ],
  ]
}

function createPaddingRules(): Rule[] {
  return [
    [/^ex-pl-(\d+)$/, match => ({ padding: `${match[1]}px ${match[1]}px ${match[1]}px 0` })],
    [/^ex-pr-(\d+)$/, match => ({ padding: `${match[1]}px 0 ${match[1]}px ${match[1]}px` })],
    [/^ex-pt-(\d+)$/, match => ({ padding: `0 ${match[1]}px ${match[1]}px ${match[1]}px` })],
    [/^ex-pb-(\d+)$/, match => ({ padding: `${match[1]}px ${match[1]}px 0 ${match[1]}px` })],
  ]
}

function createBorderRules(): Rule[] {
  return [
    [
      /^bd-(\d+)-(top|left|bottom|right)-(dashed|solid)-(\w+)$/,
      match => ({
        [`border-${match[2]}`]: `${match[1]}px ${match[3]} #${match[4]}`,
      }),
    ],
  ]
}

function textRules(): Rule[] {
  return [
    [
      /text-none-select/,
      () => ({
        '-webkit-touch-callout': 'none',
        '-webkit-user-select': 'none',
        '-khtml-user-select': 'none',
        '-moz-user-select': 'none',
        '-ms-user-select': 'none',
        'user-select': 'none',
      }),
    ],
    [
      /text-ellipsis-(\d+)/,
      match => ({
        'overflow': 'hidden',
        'text-overflow': 'ellipsis',
        'display': '-webkit-box',
        '-webkit-line-clamp': match[1],
        '-webkit-box-orient': 'vertical',
      }),
    ],
  ]
}

function safeRules(): Rule[] {
  return [
    [
      /^([pm]|gap)([trblxy]?)-safe$/,
      ([, prop, dir]) => {
        const baseMap: Record<string, string> = {
          p: 'padding',
          m: 'margin',
          gap: 'gap',
        }
        const base = baseMap[prop]

        const dirMap: Record<string, string> = {
          t: 'top',
          r: 'right',
          b: 'bottom',
          l: 'left',
        }

        if (!dir) {
          // 没写方向 → 四边全加
          return {
            [`${base}-top`]: `env(safe-area-inset-top)`,
            [`${base}-right`]: `env(safe-area-inset-right)`,
            [`${base}-bottom`]: `env(safe-area-inset-bottom)`,
            [`${base}-left`]: `env(safe-area-inset-left)`,
          }
        }

        if (dirMap[dir]) {
          return {
            [`${base}-${dirMap[dir]}`]: `env(safe-area-inset-${dirMap[dir]})`,
          }
        }

        if (dir === 'x') {
          return {
            [`${base}-left`]: `env(safe-area-inset-left)`,
            [`${base}-right`]: `env(safe-area-inset-right)`,
          }
        }

        if (dir === 'y') {
          return {
            [`${base}-top`]: `env(safe-area-inset-top)`,
            [`${base}-bottom`]: `env(safe-area-inset-bottom)`,
          }
        }
      },
    ],

    /**
     * 全称：p-top-safe / m-bottom-safe
     */
    [
      /^(p|m|gap)-(top|right|bottom|left)-safe$/,
      ([, prop, dir]) => {
        const baseMap: Record<string, string> = {
          p: 'padding',
          m: 'margin',
          gap: 'gap',
        }
        return {
          [`${baseMap[prop]}-${dir}`]: `env(safe-area-inset-${dir})`,
        }
      },
    ],

    /**
     * 定位：top-safe / left-safe ...
     */
    [
      /^(top|right|bottom|left)-safe$/,
      ([, dir]) => {
        return { [dir]: `env(safe-area-inset-${dir})` }
      },
    ],
  ]
}

export default defineConfig({
  presets: [
    presetUni(),
    presetIcons({
      scale: 1.2,
      warn: true,
      extraProperties: {
        'display': 'inline-block',
        'vertical-align': 'middle',
      },
      // HBuilderX 必须针对要使用的 Collections 做异步导入
      // collections: {
      //   carbon: () => import('@iconify-json/carbon/icons.json').then(i => i.default),
      // },
    }),
  ],
  rules: [
    ...createSizeRules(),
    ...createFlexRules(),
    ...createPositionRules(),
    ...createPaddingRules(),
    ...createBorderRules(),
    ...textRules(),
    ...safeRules(),
  ],
  transformers: [
    transformerDirectives(),
    transformerVariantGroup(),
  ],
})
