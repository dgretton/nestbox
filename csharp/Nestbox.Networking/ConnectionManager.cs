using System;
using Nestbox.Interfaces;

namespace Nestbox.Networking
{
    public class ConnectionManager
    {
        public IConnection CreateConnection(ConnectionConfig config)
        {
            switch (config.Type.ToLower())
            {
                //case "tls":
                //    return new TlsConnection(config.Ip, config.Port, config.CertPath, config.KeyPath);
                case "tcp":
                    return new TcpConnection(config.Ip, config.Port);
                default:
                    throw new ArgumentException($"Unsupported connection type: {config.Type}");
            }
        }
    }
}
