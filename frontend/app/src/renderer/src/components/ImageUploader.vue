<template>
  <v-col cols="4">
    <v-card>
      <v-card-title>Image Upload Preview</v-card-title>
      <v-card-text>
        <v-file-input
          label="Upload Image"
          v-model="files"
          accept="image/*"
          clearable
        ></v-file-input>
        <div v-if="imageStore.imageUrl" class="mt-3 preview-container">
          <v-img :src="imageStore.imageUrl"></v-img>
          <v-btn color="primary" class="mt-3 recommendation-btn" @click="imageStore.predictOutfit">
            Recommend Outfit
          </v-btn>
        </div>
      </v-card-text>
    </v-card>
  </v-col>
</template>

<script setup>
import { ref, watch } from 'vue'
import { useImageStore } from '../stores/ImageStore'

const files = ref([])
const imageStore = useImageStore()
// const imageUrl = ref('');

watch(files, (newFiles) => {
  const file = newFiles[0] //
  if (file && file instanceof Blob) {
    imageStore.setImageBlob(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      imageStore.setImageUrl(e.target.result)
    }
    reader.readAsDataURL(file)
  } else {
    imageStore.setImageUrl('')
    imageStore.setImageBlob(null)
    imageStore.setPredictedImageUrl('')
  }
})
</script>

<style scoped>
.preview-container {
  display: flex;
  flex-direction: column;
  /* align-items: center; */
}

.recommendation-btn {
  width: 60%;
  align-self: center;
}
</style>
