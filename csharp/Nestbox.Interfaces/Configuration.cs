using System.Collections.Generic;

namespace Nestbox.Interfaces
{
    public interface IConfigurationProvider
    {
        AppConfig LoadConfig(string filePath);
    }

    public class NetworkConfig
    {
        public string DefaultConnection { get; set; }
        public Dictionary<string, ConnectionConfig> Connections { get; set; }
    }

    public class ConnectionConfig
    {
        public string Type { get; set; }
        public string Ip { get; set; }
        public int Port { get; set; }
        public string CertPath { get; set; }
        public string KeyPath { get; set; }
    }

    public class OptimizerConfig
    {
        public string Type { get; set; }
        public float LearningRate { get; set; }
    }

    public class AppConfig
    {
        public NetworkConfig Network { get; set; }
        public OptimizerConfig Optimizer { get; set; }
    }
}