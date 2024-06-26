namespace Nestbox.Interfaces
{
    public interface IConnection
    {
        void Connect();
        void Send(byte[] data);
        bool IsConnected();
        void Close();
    }
}
