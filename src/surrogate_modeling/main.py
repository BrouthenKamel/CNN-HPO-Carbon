from architectures.alexnet import CustomAlexNet
from torchsummary import summary
from trainers.alexnet.mnist import run_mnist

input_path = "./datasets/generated/alexnet_random_configs_v5.csv"
output_path = "./datasets/accus/alexnet_mnist_v5.csv"

# model = CustomAlexNet("./datasets/generated/alexnet_random_configs.csv", row=1, input_channels=3)

# summary(model=model, input_size=(3,224,224))

for i in range(200):
    run_mnist(input_path, i, output_path, device="cuda")