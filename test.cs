using System
using System.Net.Sockets;

class Test
{
  protected class SendFeedback(int res)
  {
    using (TcpClient client = new TcpClient("127.0.0.1", 5555))
    {
      NetworkStream stream = client.GetStream();

      byte[] data = BitConverter.GetBytes(res);
      stream.Write(data, 0, data.Length);
    }
  }

  public static void Main()
  {
    Console.WriteLine("Send test feedback");
    SendFeedback(42);
  }
}