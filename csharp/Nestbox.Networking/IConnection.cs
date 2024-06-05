namespace Nestbox.Networking
{
    public interface IConnection
    {
        void Connect();
        void Send(byte[] data);
        void Close();
    }
}
