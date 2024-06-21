# 3D Reconstruction

In this assignment, we are provided two images of a temple from slightly different angles, and use this to generate a point cloud and depth/disparity maps. We utilize triangulation 
and the eight-point algorithm to do so. 

This first pair of image shows the result of a correct eight-point algorithm on the pair of images. Clicking on a point in one image should result in the corresponding point on the other image being on the epipolar line, and we can see this is indeed the case. 

<img width="253" alt="Screenshot 2024-06-21 at 12 47 08 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/98280b46-61e6-44bf-ae5e-f294d42c5f97">
<img width="255" alt="Screenshot 2024-06-21 at 12 47 22 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/cf1a763d-c604-449d-a1fe-f012bebb6baa">

#### 3D reconstruction from multiple angles
<img width="243" alt="Screenshot 2024-06-21 at 12 49 37 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/3b926dc3-f15d-4057-b232-bf94823f0ca0">
<img width="253" alt="Screenshot 2024-06-21 at 12 49 45 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/e1bfaf72-e69c-4ca5-af0e-968a5481fc1e">
<img width="253" alt="Screenshot 2024-06-21 at 12 49 59 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/dc4d7116-9d0e-4453-a535-1c8b49280bae">

#### Disparity and Depth maps, respectively
<img width="284" alt="Screenshot 2024-06-21 at 12 52 07 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/d4def0d0-b260-4b91-b61e-6e32c1597f34">
<img width="288" alt="Screenshot 2024-06-21 at 12 52 14 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/ab99ca52-8c4f-4d2d-ba76-3a5866a1f118">
