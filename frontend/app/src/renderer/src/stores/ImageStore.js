import { defineStore } from 'pinia';

export const useImageStore = defineStore('image', {
  state: () => ({
    imageBlob: null,
    imageUrl: '',
    predictedImageUrl: '',
  }),
  actions: {
    setImageUrl(url) {
      this.imageUrl = url;
    },
    setImageBlob(fileBlob) {
      this.imageBlob = fileBlob;
    },
    setPredictedImageUrl(url) {
      this.predictedImageUrl = url;
    },
    async predictOutfit() {
      if (!this.imageUrl) {
        alert("Please upload an image first.");
        return;
      }

      try {
        const formData = new FormData();
        // Assuming the image is already in the form of a File or Blob
        formData.append('file', this.imageBlob) // Adjust if necessary

        const response = await fetch('http://127.0.0.1:8000/predict/', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.blob();
        this.predictedImageUrl = URL.createObjectURL(data);
      } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction.');
      }
    }
  },
});
