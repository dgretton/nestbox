//for ArgumentException
using System;

namespace Nestbox.Networking
{
    public class ConnectionManager
    {
        public IConnection CreateConnection(string type, string ipAddress, int port)
        {
            if (type == "TCP")
            {
                return new TcpConnection(ipAddress, port);
            }
            else
            {
                throw new ArgumentException("Unsupported connection type");
            }
        }
    }
}
