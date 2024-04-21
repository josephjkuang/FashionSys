<h1 align="center">
  FashionSys
  </br>
</h1>

<table align="center">
  <tr>
    <td align="center"><a height="75px;" alt="Joseph Kuang"/><br /><b>Joseph Kuang</b></a><br /></td>
    <td align="center"><a width="75px;" alt="Xinshuo Lei"/><br /><b>Xinshuo Lei</b></a><br /></td>
    <td align="center"><a width="75px;" alt="Hari Umesh"/><br /><b>Hari Umesh</b></a><br /></td>
    </tr>
</table>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#repository-breakdown">Repository Breakdown</a> •
  <a href="#technologies">Technologies</a> •
  <a href="#running-the-project">Running The Project</a> •
</p>

## Abstract

Privacy concerns in fashion technology have increasingly become a paramount issue, leaving users vulnerable to data leakages due to the oversight of most fashion tools. This paper introduces \name, a novel system designed to tackle these concerns by providing outfit completion within a privacy-oriented framework. \name leverages edge computing principles to process all operations involving identifiable user data, such as personal clothing images and fashion biases, locally. It utilizes an image inferencing tool to extract low-level embeddings at the user-side, eliminating the need to transmit user clothing data to global servers. Furthermore, \name integrates a filtering mechanism based on user preferences at the local level to enhance suggested outfits according to implicit and explicit feedback. Through our experiments, we demonstrate that our system can achieve these goals with less than 100 MB of storage overhead and an improved scalability rate for additional client requests.

## Repository Breakdown

- TODO

## Technologies

- TODO

## Running the Project

- Download embedding files from https://drive.google.com/drive/folders/1gz4woiYTK0COxMHfgwP9KSsQ9z0zcY_q?usp=sharing
- Download images from https://www.kaggle.com/datasets/dnepozitek/polyvore-outfits/data

- Run cells in `outfit_generation/client.ipynb`
- Run cells in `outfit_generation/server.ipynb`

