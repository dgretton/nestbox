using System;
using System.IO;
using System.Collections.Generic;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace Nestbox.Core
{
    public class ConfigLoader
    {
        public static AppConfig LoadConfig(string path)
        {
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build();

            using (var reader = new StreamReader(path))
            {
                var config = deserializer.Deserialize<AppConfig>(reader);
                return config;
            }
        }
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
