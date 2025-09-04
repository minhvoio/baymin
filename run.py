from bni_netica.support_tools import get_nets, printNet

def main():
    print()
    print("Hi, my name is Bayesianista. I can help you work with Bayesian Networks.\n")
    nets = get_nets()

    
    for i, net in enumerate(nets):
        print(f"{i}: {net.name()}")

    print()
    choice = int(input("Enter the number of the network you want to use: "))
    print()
    if choice < 0 or choice >= len(nets):
        print("Invalid choice. Exiting.")
        return
    
    net = nets[choice]
    print(f"You chose: {net.name()}")
    printNet(net)
    print()

if __name__ == "__main__":
    main()
