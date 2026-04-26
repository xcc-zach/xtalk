<script setup lang="ts">
import { Base64 } from 'js-base64'

definePage({
  layout: false,
  style: { navigationStyle: 'custom' },
})

const url = ref('')

onLoad((options) => {
  // console.log(Base64)
  if (options!.url) {
    url.value = Base64.decode(options!.url)
    // console.log('✅ WebView URL:', url.value)
  }
  else {
    uni.showToast({ title: '参数错误', icon: 'none' })
  }
})
</script>

<template>
  <view class="webview-container">
    <web-view v-if="url" :src="url" class="webview" />
  </view>
</template>

<style scoped>
.webview-container {
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100vh;
}
.webview-title {
  background: #007aff;
  color: #fff;
  text-align: center;
  padding: 10rpx;
}
.webview {
  flex: 1;
}
</style>
