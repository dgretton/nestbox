using System;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using Nestbox.Interfaces;

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
}
