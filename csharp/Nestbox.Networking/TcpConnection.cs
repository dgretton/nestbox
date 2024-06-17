using System;
using System.Net.Sockets;
using System.Text;
using Nestbox.Interfaces;

namespace Nestbox.Networking
{
    public class TcpConnection : IConnection
    {
        private TcpClient _client;
        private NetworkStream _stream;
        private string _ipAddress;
        private int _port;

        public TcpConnection(string ipAddress, int port)
        {
            _ipAddress = ipAddress ?? throw new ArgumentNullException("IP address cannot be null");
            _port = port > 0 ? port : throw new ArgumentException("Port must be greater than 0");
        }

        public void Connect()
        {
            _client = new TcpClient(_ipAddress, _port);
            _stream = _client.GetStream();
        }

        public void Send(byte[] data)
        {
            if (_stream.CanWrite)
            {
                _stream.Write(data ?? throw new ArgumentNullException("Data cannot be null"), 0, data.Length);
            }
        }

        public bool IsConnected()
        {
            return _client != null && _client.Connected;
        }

        public void Close()
        {
            _stream?.Close();
            _client?.Close();
        }
    }
}
