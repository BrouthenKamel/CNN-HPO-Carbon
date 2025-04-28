from architectures.alexnet import CustomAlexNet
from torchsummary import summary
from trainers.alexnet.mnist import run_mnist

input_path = "./datasets/generated/alexnet_random_configs.csv"
output_path = "./datasets/accus/alexnet_mnist.csv"

model = CustomAlexNet("./datasets/generated/alexnet_random_configs.csv", row=1, input_channels=3)

summary(model=model, input_size=(3,224,224))

# run_mnist(input_path, 0, output_path, device="cuda")