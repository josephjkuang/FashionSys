<script setup lang="ts">
// import Versions from './components/Versions.vue'
import ImageUploader from './components/ImageUploader.vue'
import OutfitPrediction from './components/OutfitPrediction.vue'

const ipcHandle = () => window.electron.ipcRenderer.send('ping')
</script>

<template>
  <div id="container">
    <v-row>
      <image-uploader />
      <outfit-prediction />
    </v-row>
  </div>
</template>

<style scoped>
#container {
  padding: 20px;
}
</style>