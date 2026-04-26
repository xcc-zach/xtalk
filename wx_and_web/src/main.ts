import uviewPlus from 'uview-plus'
import { createSSRApp } from 'vue'
import App from './App.vue'
import 'uno.css'
import './assets/style/index.css'

export function createApp() {
  const app = createSSRApp(App)
  app.use(uviewPlus)
  return {
    app,
  }
}
