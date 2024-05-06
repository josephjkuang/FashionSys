<h1 align="center">
  PrivéStyler
  </br>
</h1>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#repository-breakdown">Repository Breakdown</a> •
  <a href="#running-the-project">Running The Project</a> •
</p>

## Abstract

Privacy concerns in fashion technology have increasingly become a paramount issue, leaving users vulnerable to data leakages due to the oversight of most fashion tools. This paper introduces PrivéStyler, a novel system designed to tackle these concerns by providing outfit completion within a privacy-oriented framework. PrivéStyler leverages edge computing principles to process all operations involving identifiable user data, such as personal clothing images and fashion biases, locally. It employs an image inference tool to extract low-level embeddings at the user end, which are further perturbed with a form of Laplacian noise referred to as cluster-centric noise. By doing so, PrivéStyler eliminates the need to transmit user images to global servers and reduces the risk of data reconstruction. Our experiments demonstrate that our system can achieve this with only $209.8$ MB of storage overhead for the lowest-tier and maintains a Spearman Rank Correlation Coefficient of 0.905 with an $\epsilon = 6.98$ for privacy loss.

## Repository Breakdown

- `backend-api` includes the backend application for the Electron app
- `data-processing` includes helper scripts for when we were normalizing the data
- `frontend` includes the code frontend code for the application
- `measurements` includes code for end-to-end system metric evaluations. Additional instructions provided inside
- `outfit_generation` includes the main points for quick reproducing the results of our paper
  - `Branch: evaluation` includes additional scripts for evaluating the effects of cluster-centric noise
- `polyvore_outfits` includes all the images, embeddings, and metadata


## Running the Project

- Download embedding and metadata files in the `polyvore_outfits` folder: https://drive.google.com/drive/folders/1gz4woiYTK0COxMHfgwP9KSsQ9z0zcY_q?usp=sharing
- Download images from and place inside of `polyvore_outfits/images` https://www.kaggle.com/datasets/dnepozitek/polyvore-outfits/data

- For quick reproduce of client side: Run cells in `outfit_generation/client.ipynb`
- For quick reproduce of encryption: Run cells in `Encryption.ipynb`
- For quick reproduce of client side: Run cells in `outfit_generation/server.ipynb`
- For end-to-end reproduce: Follow instructions in `measurements/readme.md`

