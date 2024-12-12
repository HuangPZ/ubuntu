from scapy.all import sniff, IP, UDP

# Define the interface to listen on
interface = "enp4s0"

# Define a callback function to process received packets
def packet_callback(packet):
    if IP in packet and UDP in packet:
        print(f"Packet received:")
        print(f"  Source IP: {packet[IP].src}")
        print(f"  Destination IP: {packet[IP].dst}")
        print(f"  Payload: {packet[UDP].payload}")

# Start sniffing on the specified interface
print(f"Listening on {interface}...")
sniff(iface=interface, filter="udp", prn=packet_callback)
