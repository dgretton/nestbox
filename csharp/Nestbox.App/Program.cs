using System;
using Google.Protobuf;
using Nestbox.Protos;
using Nestbox.Networking; // Ensure this using directive is correct based on your namespaces

namespace Nestbox.App
{
    class Program
    {
        static void Main(string[] args)
        {
            ConnectionManager manager = new ConnectionManager();
            string ipAddress = "127.0.0.1";
            int port = 12345;
            IConnection connection = manager.CreateConnection("TCP", ipAddress, port);

            try
            {
                Console.WriteLine("Connecting to server at " + ipAddress + " on port " + port + "...");
                connection.Connect();
                Console.WriteLine("Connected to server. Sending message...");

                string message = "Hello, World!";
                byte[] data = System.Text.Encoding.UTF8.GetBytes(message);
                connection.Send(data);

                Console.WriteLine("Message sent!");
                SendTwig(connection);
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
                Measurements =
                {
                    new Measurement
                    {
                        Mean = { 1.0f, 2.0f, 3.0f },
                        Covariance = new CovarianceMatrix { UpperTriangle = { 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f } },
                        Transform = new TransformationMatrix { Data = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f } }
                    }
                }
            };

            byte[] data = twig.ToByteArray();
            connection.Send(data);
        }
    }
}

