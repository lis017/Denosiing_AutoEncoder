from MNISTData import MNISTData
from AutoEncoder import AutoEncoder


if __name__ == "__main__":
    print("Hi. I am an Auto Encoder Trainer.")
    batch_size = 32
    num_epochs = 5
    data_loader = MNISTData()
    data_loader.load_data()
    x_train = data_loader.x_train
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    auto_encoder.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=num_epochs)

    save_path = "./model/ae_model.weights.h5"
    auto_encoder.save_weights(save_path)
    print("load model weights from %s" % save_path)