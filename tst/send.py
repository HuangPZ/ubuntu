from scapy.all import sendp, Ether, IP, UDP

# Define the network interface and payload
interface = "enp4s0"
payload = b"Hello via enp4s0!"

# Build the packet
packet = Ether() / IP(src="172.31.43.45", dst="172.31.43.45") / UDP(dport=12345) / payload

# Send the packet via the specific interface
sendp(packet, iface=interface)

print(f"Packet sent via interface {interface}")

