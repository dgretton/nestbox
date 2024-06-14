using System;
using System.IO;
using Nestbox.Interfaces;
#if USE_YAML
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
#endif
#if USE_JSON
using System.Text.Json;
#endif

public class ConfigProvider
{
    public static IConfigurationProvider GetProvider(string configType)
    {
        switch (configType.ToLower())
        {
#if USE_YAML
            case "yaml":
            case "yml":
                return new YamlConfigProvider();
#endif
#if USE_JSON
            case "json":
                return new JsonConfigProvider();
#endif
            // TODO: add environment variable configuration option
            default:
                throw new ArgumentException("Unsupported config type");
        }
    }
}

#if USE_YAML
public class YamlConfigProvider : IConfigurationProvider
{
    public YamlConfigProvider()
    {
    }

    public AppConfig LoadConfig(string filePath)
    {
        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();

        using (var reader = new StreamReader(filePath))
        {
            var config = deserializer.Deserialize<AppConfig>(reader);
            return config;
        }
    }
}
#endif

#if USE_JSON
public class JsonConfigProvider : IConfigurationProvider
{
    public JsonConfigProvider()
    {
    }

    public AppConfig LoadConfig(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Configuration file not found: {filePath}");
        }

        string json = File.ReadAllText(filePath);
        return JsonSerializer.Deserialize<AppConfig>(json);
    }
}
#endif
