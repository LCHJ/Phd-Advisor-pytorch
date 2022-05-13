# 2022��4��28��

usps

1. ����eval�����ݼ����Ͻ�

   best_acc =[76.925, 81.1, 73.55]
   best_nmi =[79.64592, 86.8694, 73.85137]
2. ����epoch size = 1000

   best_acc =[76.08088, 81.38309, 73.7578]
   best_nmi =[77.22623, 86.91896, 72.79656]
3. ʹ��ԭ����resnet.py

   best_acc =[79.05, 81.36667, 74.31667]
   best_nmi =[80.38201, 87.78045, 75.18109]
4. self.fc_hidden2=256\
5. relu-bn, cov1+relu
6. reparamaterise

best_acc =[76.14541, 66.93913, 72.40267]
best_nmi =[78.46502, 80.73548, 70.74635]

best_acc =[79.17832, 81.13573, 75.0484]
best_nmi =[82.93977, 87.41163, 75.96749]

./logs/USPS/Rho-0.7_gama-0.01_mse_y-decay_line2_lr0.0001

best_acc =[77.65111, 81.2863, 74.66122]
best_nmi =[80.26383, 86.96592, 75.17476]

# umist_1024

./logs/umist_1024/Rho-0.7_gama-0.02_mse_line2_mu

best_acc =[79.47826, 73.21739, 57.56522]
best_nmi =[92.14501, 86.52953, 73.79124]

acc=[0.91826, 0.71652, 0.52348]     nmi=[0.94184, 0.88653, 0.72489]

./logs/umist_1024/Rho-0.7_gama-0.005_mse_line2_mu

best_acc =[91.47826, 70.6087, 53.21739]
best_nmi =[95.15643, 87.52629, 75.1391]

**./logs/umist_1024/Rho-0.7_gama-0.01_mse_line2_mu_300**

**best_acc =[92.34783, 74.78261, 53.21739]
best_nmi =[95.65705, 86.78584, 73.24269]**

./logs/umist_1024/Rho-0.7_gama-0.008_mse_line2_mu

best_acc =[85.73913, 71.47826, 57.04348]
best_nmi =[93.301, 87.46183, 74.93806]

./logs/umist_1024/AHCL-kl0.7_1_0.01

best_acc =[91.47826, 60.52174, 45.91304]
best_nmi =[95.63699, 75.52361, 65.92921]

./logs/umist_1024/Rho-0.7_gama-0.01_mse_line2_mu_500

best_acc =[86.43478, 55.30435, 43.30435]
best_nmi =[94.23822, 75.29651, 63.81759]


---

# **COIL20**

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu ===��Start Training��=== Epoch 401/401:
100%|������������������������������������������������������������������| 20/20 [00:09<00:00, 2.05it/s, A_max=199,
A_mean=140, ---��Start Validating��---
>>-- y z_sp z_kmeans --<<
~~best_acc =[71.38889, 81.66667, 79.23611]
best_nmi =[86.54779, 90.5656, 85.70076]~~

---

2022��5��4��

COIL ./logs/COIL20/Rho-0.7_gama-0.02_mse_line2_mu_relu

�� = 0.7�� �� = 0.02

~~epoch size = 100~~

~~best_acc =[**75.13889**, 79.79167, 78.19444]
best_nmi =[**92.11986,** 89.65494, 86.09488]~~

---

**epoch size = 200**

best_acc =[**81.38889, 83.26389,** 75.625]
best_nmi =[**92.76465, 92.52305**, 83.57469]

./logs/COIL20/Rho-0.7_gama-0.02_mse_line2_mu_200_resnet

~~best_acc =[75.48611, 77.43056, 76.38889]
best_nmi =[90.8901, 89.87812, 84.42356]~~


---

\./logs/COIL20/Rho-0.7_gama-0.02_mse_line2_mu_200_resnet

~~best_acc =[78.40278, 77.43056, 76.38889]
best_nmi =[91.47903, 89.87812, 84.42356]~~

---

~~epoch size = 400~~

~~best_acc =[73.40278, 79.86111, 76.11111]
best_nmi =[86.80945, 89.42315, 83.83811]~~


---

~~epoch size = 800~~

~~best_acc =[72.01389, 83.33333, 75.90278]~~

~~best_nmi =[84.96934, 93.00544, 85.02037]~~

---

~~epoch size = 1440~~

~~best_acc =[71.80556, 84.23611, 79.30556]
best_nmi =[86.03292, 91.03292, 85.4977]~~



> ./logs/COIL20/Rho-0.7_gama-0.01_mse_y-decay_line2_lr0.0004
>
> -- y   z_sp    z_kmeans --<<
> best_acc =[70.55556, 83.40278, 72.63889]
> best_nmi =[81.78782, 92.73519, 82.57324]
>
> ./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu
>
> best_acc =[82.5, 83.68056, 77.08333]
> best_nmi =[94.70433, 93.04356, 85.41385]
>
> ./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_2
>
> best_acc =[72.98611, 82.84722, 74.44444]
> best_nmi =[89.68778, 91.6369, 84.09908]

~~./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_relu s2-s1~~

~~best_acc =[77.29167, 83.125, 77.15278]
best_nmi =[91.91132, 92.73495, 86.03067]~~

COIL20

�� = 0.7�� �� = 0.01

epoch size = 100

~~best_acc =[76.875, 77.84722, 75.97222]
best_nmi =[92.32874, 91.86046, 85.03384]~~

---

epoch size = 200

~~best_acc =[79.72222, 81.875, 77.56944]
best_nmi =[92.27918, 92.06055, 85.09479]~~

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_200_resnet

best_acc =[81.52778, 83.33333, 76.18056]
best_nmi =[93.19611, 92.85171, 84.23205]

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_relu_200_resnet

**best_acc =[82.15278, 82.29167, 75.90278]
best_nmi =[93.477, 90.92294, 84.04559]**

~~./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_200_resnet2~~

~~best_acc =[76.73611, 79.23611, 75.20833]
best_nmi =[91.73691, 92.81804, 84.69843]~~

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_200_resnet_nodecay

best_acc =[82.5, 85.69444, 76.52778]
best_nmi =[94.02657, 92.23605, 84.22218]

~~./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_200_resnet_nodecay-1e-5~~

~~best_acc =[76.52778, 82.63889, 75.06944]
best_nmi =[86.58157, 92.02758, 82.70511]~~

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_200

best_acc =[82.29167, 83.26389, 77.08333]
best_nmi =[95.01982, 92.59707, 85.77999]

**./logs/COIL20/AHCL0.7_1_0.012_45**

**best_acc =[88.68056, 82.22222, 77.01389]
best_nmi =[96.59352, 92.45135, 83.77921]**

**./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_1000**

**best_acc =[87.70833, 83.81944, 76.94444]
best_nmi =[96.52018, 93.36794, 84.72848]**

./logs/COIL20/Rho-0.7_gama-0.01_mse_line2_mu_1000 no_decay-Y

best_acc =[89.93056, 81.31944, 74.16667]
best_nmi =[96.71796, 91.37435, 83.58025]

---

~~�� = 0.7�� �� = 0.03~~

~~epoch size = 200~~

~~./logs/COIL20/Rho-0.7_gama-0.03_mse_line2_mu_relu_200~~

~~best_acc =[76.25, 82.77778, 76.45833]
best_nmi =[91.5499, 91.85129, 84.88383]~~

---

~~�� = 0.8�� �� = 0.01~~

~~epoch size = 200~~

~~./logs/COIL20/Rho-0.8_gama-0.01_mse_line2_mu_relu_200_resnet~~

~~best_acc =[69.16667, 79.58333, 77.29167]
best_nmi =[83.33724, 93.54924, 85.95363]~~

~~./logs/COIL20/Rho-0.8_gama-0.02_mse_line2_mu_relu_200_resnet~~

~~best_acc =[67.91667, 83.19444, 76.80556]
best_nmi =[82.79264, 90.99109, 86.09559]~~

---

~~�� = 0.6�� �� = 0.01~~

~~epoch size = 200~~

~~./logs/COIL20/Rho-0.6_gama-0.01_mse_line2_mu_relu_200_resnet~~

~~best_acc =[75.20833, 83.19444, 76.18056]
best_nmi =[90.4139, 92.68066, 85.14118]~~

~~�� = 0.6�� �� = 0.02~~

~~./logs/COIL20/Rho-0.6_gama-0.02_mse_line2_mu_relu_200_resnet~~

~~best_acc =[69.09722, 82.5, 77.5]
best_nmi =[85.88067, 91.70632, 84.36766]~~

---

# fashion

./logs/fashion-mnist/Rho-0.76_gama-0.01_mse_line2_mu_200_resnet

best_acc =[59.44811, 59.59808, 68.35633]
best_nmi =[63.38183, 62.49232, 63.71331]
