using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Google.Protobuf;
using Nestbox.Protos;
using Nestbox.Networking;
using Nestbox.Interfaces;

namespace Nestbox.App
{
    class Program
    {
        static void Main(string[] args)
        {
            //var configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config", "example_config.yaml"); //TODO: get from cmd args and only use config/config.yaml as default            
            //config-path is the name of the config file arg
            //'string[]' does not contain a definition for 'FirstOrDefault'
            //IEnumerable<string> argsIter = args;
            //string configPath = argsIter.FirstOrDefault(arg => arg.StartsWith("config-path="))?.Split('=')[1] ?? "config/config.yaml";
            //using cast instead
            string configPath = args.FirstOrDefault(arg => arg.StartsWith("config-path=") || arg.StartsWith("--config-path="))?.Split('=')[1] ?? "config/config.yaml";
            //string configPath = args.FirstOrDefault(arg => arg.StartsWith("config-path="))?.Split('=')[1] ?? "config/config.yaml";
            if (!File.Exists(configPath))
            {
                Console.WriteLine("Config file not found at " + configPath);
                return;
            }
            Console.WriteLine("Config file:" + configPath);
            IConfigurationProvider configProvider = ConfigProvider.GetProvider("yaml");
            AppConfig config = configProvider.LoadConfig(configPath);
            Console.WriteLine("Config loaded.");
            //var connectionConfig = config.Network.Connections[config.Network.DefaultConnection]; //TODO: use cmd arg to give connection name if available
            string connectionName = args.FirstOrDefault(arg => arg.StartsWith("connection="))?.Split('=')[1] ?? config.Network.DefaultConnection;
            Console.WriteLine("Using connection config name: " + connectionName);
            var connectionConfig = config.Network.Connections[connectionName];

            ConnectionManager manager = new ConnectionManager();

            // string ipAddress = "127.0.0.1";
            // int port = 12345;
            string ipAddress = connectionConfig.Ip;
            int port = connectionConfig.Port;
            string type = connectionConfig.Type;
            Console.WriteLine("From config file " + configPath + ":");
            Console.WriteLine("Connection type: " + type);
            Console.WriteLine("Connection IP: " + ipAddress);
            Console.WriteLine("Connection port: " + port);

            IConnection connection = manager.CreateConnection(connectionConfig);
            // IConnection connection = manager.CreateConnection("TCP", ipAddress, port);


            try
            {
                Console.WriteLine("Connecting to server at " + ipAddress + " on port " + port + "...");
                connection.Connect();
                Console.WriteLine("Connected to server. Sending message...");

                // string message = "Hello, World!";
                // byte[] data = System.Text.Encoding.UTF8.GetBytes(message);
                // connection.Send(data);

                // Console.WriteLine("Message sent!");
                SendTwig(connection);
                Console.WriteLine("Twig sent!");
            }
            catch (Exception e)
            {
                Console.WriteLine("An error occurred: " + e.Message);
            }
            finally
            {
                connection.Close();
                Console.WriteLine("Connection closed.");
            }
        }


        public static void SendTwig(IConnection connection)
        {
            var twig = new Twig
            {
                CoordSysId = "c05d4581-f376-4c21-b722-b7b7694a9fd2",
                StreamId = "xxxx",
                Measurements = {
                    new MeasurementSet
                    {
                        Dimensions = { Dimension.X, Dimension.Y, Dimension.Z },
                        Samples = {
                            new Sample
                            {
                                Mean = { 1.0f, 2.0f, 3.0f },
                                Covariance = new CovarianceMatrix { UpperTriangle = { 0.03f, 0.0f, 0.03f, 0.0f, 0.0f, 0.03f } }
                            }
                        },
                        Transform = new TransformationMatrix { Data = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f } },
                        IsHomogeneous = false
                    }
                }
            };

            byte[] data = twig.ToByteArray();
            byte[] lengthPrefix = BitConverter.GetBytes(data.Length);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(lengthPrefix);
            }
            connection.Send(lengthPrefix);
            connection.Send(data);
        }
    }
}

