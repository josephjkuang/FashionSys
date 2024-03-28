<template>
  <v-col cols="8"  v-if="imageStore.predictionInProgress">
    <v-card>
      <v-card-title>
        suggesting outfit
        <v-progress-circular indeterminate class="ml-5" :size="25"></v-progress-circular>
      </v-card-title>
    </v-card>
  </v-col>
  <v-col cols="8"  v-if="imageStore.predictedImageUrl">
    <v-card>
      <v-card-title>Suggested Outfit</v-card-title>
      <v-card-text>
        <div class="mt-3">
          <v-img :src="imageStore.predictedImageUrl"  aspect-ratio="1.4"></v-img>
        </div>
      </v-card-text>
      <div class="feedback-container">
        <v-btn icon="mdi-thumb-up-outline" variant="text"/>
        <v-btn icon="mdi-thumb-down-outline" variant="text"/>
      </div>
    </v-card>
  </v-col>
</template>

<script setup>
import { useImageStore } from '../stores/ImageStore'

const imageStore = useImageStore()
</script>

<style scoped>
.feedback-container {
  display: flex;
  justify-content: center;
  /* padding-right: 10px; */
}
</style>