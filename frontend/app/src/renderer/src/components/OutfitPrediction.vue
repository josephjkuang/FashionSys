<template>
  <v-col cols="8"  v-if="imageStore.predictedImageUrl">
    <v-card>
      <v-card-title>Suggested Outfit</v-card-title>
      <v-card-text>
        <div class="mt-3">
          <v-img :src="imageStore.predictedImageUrl"  aspect-ratio="1.2"></v-img>
        </div>
      </v-card-text>
    </v-card>
  </v-col>
</template>

<script setup>
import { useImageStore } from '../stores/ImageStore'

const imageStore = useImageStore()
</script>