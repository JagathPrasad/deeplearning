using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Keras;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.Utils;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Numpy;

namespace DeepLearning.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class WeatherForecastController : ControllerBase
    {
        private readonly IWebHostEnvironment _webHostEnvironment;

        private static readonly string[] Summaries = new[]
        {
            "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
        };

        private readonly ILogger<WeatherForecastController> _logger;

        public WeatherForecastController(ILogger<WeatherForecastController> logger, IWebHostEnvironment webHostEnvironment)
        {
            _logger = logger;
            _webHostEnvironment = webHostEnvironment;
        }

        [HttpGet]
        public IEnumerable<WeatherForecast> Get()
        {
            //string webRootPath = _webHostEnvironment.WebRootPath;
            //var x = Path.Combine(_webHostEnvironment.WebRootPath, "/data/breastcancer");
            TestDrugDiscovery();
            var rng = new Random();
            return Enumerable.Range(1, 5).Select(index => new WeatherForecast
            {
                Date = DateTime.Now.AddDays(index),
                TemperatureC = rng.Next(-20, 55),
                Summary = Summaries[rng.Next(Summaries.Length)]
            }).ToArray();
        }

        public void TestXOR()
        {

            NDarray x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            NDarray y = np.array(new float[] { 0, 1, 1, 0 });

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(2)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            //Compile and train
            model.Compile(optimizer: "sgd", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(x, y, batch_size: 2, epochs: 1000, verbose: 1);

            //Save model and weights
            string json = model.ToJson();
            System.IO.File.WriteAllText("model.json", json);
            model.SaveWeight("model.h5");

            //Load model and weight
            var loaded_model = Sequential.ModelFromJson(System.IO.File.ReadAllText("model.json"));
            loaded_model.LoadWeight("model.h5");
        }


        public void TestCNN()
        {
            int batch_size = 128;
            int num_classes = 10;
            int epochs = 12;

            // input image dimensions
            int img_rows = 28, img_cols = 28;

            Shape input_shape = null;

            // the data, split between train and test sets
            var ((x_train, y_train), (x_test, y_test)) = MNIST.LoadData();

            if (Backend.ImageDataFormat() == "channels_first")
            {
                x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols);
                x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols);
                input_shape = (1, img_rows, img_cols);
            }
            else
            {
                x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1);
                x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1);
                input_shape = (img_rows, img_cols, 1);
            }

            x_train = x_train.astype(np.float32);
            x_test = x_test.astype(np.float32);
            x_train /= 255;
            x_test /= 255;
            Console.WriteLine($"x_train shape: {x_train.shape}");
            Console.WriteLine($"{x_train.shape[0]} train samples");
            Console.WriteLine($"{x_test.shape[0]} test samples");

            // convert class vectors to binary class matrices
            y_train = Util.ToCategorical(y_train, num_classes);
            y_test = Util.ToCategorical(y_test, num_classes);

            // Build CNN model
            var model = new Sequential();
            model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(),
                                    activation: "relu",
                                    input_shape: input_shape));
            model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(num_classes, activation: "softmax"));

            model.Compile(loss: "categorical_crossentropy",
                optimizer: new Adadelta(), metrics: new string[] { "accuracy" });

            model.Fit(x_train, y_train,
                        batch_size: batch_size,
                        epochs: epochs,
                        verbose: 1,
                        validation_data: new NDarray[] { x_test, y_test });
            var score = model.Evaluate(x_test, y_test, verbose: 0);
            Console.WriteLine($"Test loss: {score[0]}");
            Console.WriteLine($"Test accuracy: {score[1]}");
        }

        public void TestBreastCancer()
        {
            var batch_size = 120;
            var epoch = 25;
            var num_classes = 5;
            var image_size = 28;
            var load_date = "breast cancer images";
            /*
                         train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1./255)
            train_generator = train_datagen.flow_from_directory(
                    'data/train',
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='binary')
            validation_generator = test_datagen.flow_from_directory(
                    'data/validation',
                    target_size=(150, 150),
                    batch_size=32,
                    class_mode='binary')
            model.fit(
                    train_generator,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=validation_generator,
        validation_steps=800)
             */

            var data_gen = new Keras.PreProcessing.Image.ImageDataGenerator(rescale: 1 / 255, shear_range: 0.2f, zoom_range: 0.2f, horizontal_flip: true);
            var validate_gen = new Keras.PreProcessing.Image.ImageDataGenerator(rescale: 1 / 255);
            Tuple<int, int> tuple1 = new Tuple<int, int>(50, 50);
            string path = @"G:\Projects\deeplearning\DeepLearning\data\breastcancer";
            var train_gen = data_gen.FlowFromDirectory(path, (150, 150).ToTuple(), "grayscale", batch_size: 32, class_mode: "binary");
            var validation_gen = validate_gen.FlowFromDirectory(path, (150, 150).ToTuple(), "grayscale", batch_size: 32, class_mode: "binary");

            Tuple<int, int> tuple = new Tuple<int, int>(50, 50);
            Shape input_shape = (1000, 3, 3, 1);
            var model = new Sequential();
            model.Add(new Conv2D(32, kernel_size: (3, 3).ToTuple(), activation: "relu", input_shape: input_shape));
            model.Add(new Conv2D(64, (3, 3).ToTuple(), activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Dropout(0.25));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(num_classes, activation: "softmax"));
            var opt = new Keras.Optimizers.Adagrad(lr: 0.01f, decay: 0.01f / 100);
            string[] stringarray = new string[1];
            stringarray[0] = "accuracy";
            model.Compile(loss: "binary_crossentropy", optimizer: opt, metrics: stringarray);

            //var H = modal.Fit(x: train_gen, steps_per_epoch: 1000, // BS,
            //                  validation_data: validation_gen, validation_steps: 1000, // BS,
            //               epochs: 100);
        }


        public void TestSkinCancer()
        {
            var model = new Sequential();
            model.Add(new Conv2D(32, (3, 3).ToTuple(), input_shape: (32, 32, 3), activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: (2, 2).ToTuple()));
            model.Add(new Flatten());
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));
            string[] stringarray = new string[1];
            stringarray[0] = "accuracy";
            model.Compile(optimizer: "adam", loss: "binary_crossentropy", metrics: stringarray);
            var train_gen = new Keras.PreProcessing.Image.ImageDataGenerator(rescale: 1 / 255, shear_range: 0.2f, zoom_range: 0.2f, horizontal_flip: true);
            var test_gen = new Keras.PreProcessing.Image.ImageDataGenerator(rescale: 1 / 255);
            string path = @"G:\Projects\deeplearning\DeepLearning\data\skincancer";
            var train_set = train_gen.FlowFromDirectory(path + "\\train", target_size: (3, 3).ToTuple(), batch_size: 20, class_mode: "binary");
            var test_set = test_gen.FlowFromDirectory(path + "\\test", target_size: (3, 3).ToTuple(), batch_size: 20, class_mode: "binary");
            model.FitGenerator(train_set, steps_per_epoch: 1000, epochs: 50, validation_data: test_set, validation_steps: 400);


            //predict logic should come here.
        }


        public void TestDiabeticRetinopathy()
        {


        }

        public void TestDrugDiscovery()
        {
            var sequence_length = 100;
            var batch_size = 128;
            var epochs = 30;
            var text_path = @"G:\Projects\deeplearning\DeepLearning\data\drugdiscovery\smiles.txt";
            var vocab_path = @"G:\Projects\deeplearning\DeepLearning\data\drugdiscovery\vocab.txt";
            string text = System.IO.File.ReadAllText(text_path);
            string vocab = System.IO.File.ReadAllText(vocab_path);
            var charList = text.ToCharArray();
            var uniqueChars = vocab.ToCharArray();

            Dictionary<int, string> intToText = new Dictionary<int, string>();
            Dictionary<string, int> textToInt = new Dictionary<string, int>();
            int vocab_size = uniqueChars.Count();
            Utility.Utility utility = new Utility.Utility();
            textToInt = utility.ConvertTexttoInteger(charList);
            intToText = utility.ConvertIntegertoText(charList);

            List<Dictionary<string, int>> intList = new List<Dictionary<string, int>>();
            List<Dictionary<string, int>> intListOut = new List<Dictionary<string, int>>();
            for (int i = 0; i < (charList.Count() - sequence_length); i++)
            {
                char[] text_stringbuilder = new char[i + sequence_length];
                for (int j = 0; j < i + sequence_length; j++)
                {
                    text_stringbuilder[j] = (charList[j]);
                }
                char[] text_out = new char[i + sequence_length];
                text_out[i] = charList[i + sequence_length];
                intList.Add(utility.ConvertTexttoInteger(text_stringbuilder));
                intListOut.Add(utility.ConvertTexttoInteger(text_out));

                if (i == 100) break;
            }

            //var X = np.reshape(intList, sequence_length, 1);
            var model = new Sequential();
            model.Add(new LSTM(256));
            model.Add(new Dropout(0.2));
            model.Add(new Dense(16, activation: "softmax"));
            model.Compile(optimizer: "adam", loss: "categorical_crossentropy");

        }

        public void TestGenomeTechnology()
        {

        }
    }
}
