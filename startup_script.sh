##!/bin/bash

# Define the public key to access via ssh
public_key_data="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBFZhxTgVQUFjyMMJVGUrqoDCMNsvE7tgHWUoBIY1bxm P7group"

# Define the public and private deploy key data as strings
public_deploy_key_data="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDZOgnDjV4Naw9WQF4GNGvcFYSH9HloM79t7HFRXIMSGjDmODBVNaA7fzrUn5F5FlEqPi0++fN+zq5vGlu8TUV3F7DPPPibOBcLSGataTVJ+cOn4nkakLwnPWjT52W4gMSTS0ffPJ4qdGgvNzjIxs8NJtxAXr9F5joY6mdPD0pX7V+1mxrw4/xXhhdGrB2Uo0CUCQViu3w0VmbkXyjnFu+xPjrLcrO1Lvt68Q6A8aHlo1u4vfVOV2IRgxLdDd2bOO1dJ1D9R82SP29GViKXu/N5kX1iZGMOTtHIKbhyXST6aaZpoVuE4mfBmwL5W+0YgXJmoRhAeAkOxkdKi+tI4K06YJQBipkZVfSHNfaFArBWLp1nHDMYhFOkSe+DxqwswOyke0wEjt1yLt0r8Q5a7erktKhSZRMn+q772vt4ZcdkSVUpIQ2mqNIBP8ayZ2SEwjePTPLSr4i2BPvZI5Du1iHIB0TPknZ0daUy1h8uwXCBt5/nzCIKz5CSTLFVbl70+ZmAJ3u8x6O4Vulbt1BeZ/fWn8Ti2m0H/7dWVyTVg/IDK/Hnm+dWeVE9kCJr50TcLvJY4r8UK+/vLkNjPVHC8Qr21TMoMjcVhW60n/rY1uOFVajSOjg/FL6qLNEEAnFS9SAgSgc01tihlqUQ+K8Mxxo2UzTcqDeEeAS5ULlmR8xyyw== deploy_key_git"

private_deploy_key_data="-----BEGIN RSA PRIVATE KEY-----
MIIJJwIBAAKCAgEA2ToJw41eDWsPVkBeBjRr3BWEh/R5aDO/bexxUVyDEhow5jgw
VTWgO3861J+ReRZRKj4tPvnzfs6ubxpbvE1Fdxewzzz4mzgXC0hmrWk1SfnDp+J5
GpC8Jz1o0+dluIDEk0tH3zyeKnRoLzc4yMbPDSbcQF6/ReY6GOpnTw9KV+1ftZsa
8OP8V4YXRqwdlKNAlAkFYrt8NFZm5F8o5xbvsT46y3KztS77evEOgPGh5aNbuL31
TldiEYMS3Q3dmzjtXSdQ/UfNkj9vRlYil7vzeZF9YmRjDk7RyCm4cl0k+mmmaaFb
hOJnwZsC+VvtGIFyZqEYQHgJDsZHSovrSOCtOmCUAYqZGVX0hzX2hQKwVi6dZxwz
GIRTpEnvg8asLMDspHtMBI7dci7dK/EOWu3q5LSoUmUTJ/qu+9r7eGXHZElVKSEN
pqjSAT/GsmdkhMI3j0zy0q+ItgT72SOQ7tYhyAdEz5J2dHWlMtYfLsFwgbef58wi
Cs+QkkyxVW5e9PmZgCd7vMejuFbpW7dQXmf31p/E4tptB/+3Vlck1YPyAyvx55vn
VnlRPZAia+dE3C7yWOK/FCvv7y5DYz1RwvEK9tUzKDI3FYVutJ/62NbjhVWo0jo4
PxS+qizRBAJxUvUgIEoHNNbYoZalEPivDMcaNlM03Kg3hHgEuVC5ZkfMcssCAwEA
AQKCAgAQmkf+w0JKAkHE1Nya2szXoGllvsC8vx/FdgnfLxPxgUD1h5viF1wsz/ee
IqgKujLv6Jqqso3yxjc81KXoSXuis5PW8xEJRAs22IzIej8RbV6P6VZaOm+3DBGE
CK6UoHrlyx4uKMLp+cWwvia/6boPhkXVQaia3hLYrqnAuCl1ePYUuksf2D0EL4J9
60/DeYrJBbt8pVnD4kqw77j/mJgvUtNJk3W6xmGALwZtwooeNBMhlps2JqWfMwZx
EMs8PuidcxDUmvVSVTjbBmBUepAMXZtYttV/VXThOJcZXpch8ItAjY2iJW/Ks19R
0b3YXqRwxaVF4CbJE0yFnRYZltqGdJQdGB5VlvVf2Wu21n6maRCfaI3ezPgTsQxq
D2FAU9+VrIL9cWXsRLWFs5/KxQrwenYa1KWY3Ax4UjL1lDaGmaCjacrhe741zslx
9kF2xNmHE1P3qIRHOeq6Bxn6g6dhkF2gAvO78pcZF9TNhlg+3soArENiPNBkCLTG
Ts/evgnbmkKrHdKgfToIYyq8vip6VrAfCEGugKJBKnZQBTTemYwDjGVb9GIt0BQ+
UOZiGl6VZQgQQz2KectmTEl+4h1IB00LoxwcGuC7kKmcS6KAU/Jv28X0dPn5ibMv
joMzRYdnm2k4dgd9bsER79rUwKi3Bx2PAWP96Twi7vC3wSmU0QKCAQEA/DvQDuch
dJ/4dYsbjsc2PLaG1fUoL+w5oCWEm1TbG9FbRjYZ1kDCnbWqdGEyOum3SK7rzalS
EtnX2pa/BDCaXwgk4+YuzQnEHHaQluED0BgcWfC+kv3m6/7tZHonJRL5oR52jB5U
t2gX8whSaya9hPqX4FQY1bsPEN2FaDVLxY5D05/2JQRIDKwiz/L7Z9RhxmkggCu0
To7qkVimTh04UGIL4wh9EWAPNCzZ+rQiyY31gMKVbluY+HEzm+wrIpLuUD55GER5
ESK3oquaplHsAfTtAvv2btsA0i4wlSsx0i1eFHQz82oWc+9aJw7KRmEuV3DzqWB3
EYYgEQqEAMcpqQKCAQEA3Hhodqop5mRPB8nlvJzGNr4eaaVC04QN8mh6JUtf5dBV
atE648eVHKOGIV9NJc7cSvrRHPzmBdDuTc7I265BwXQPnSvMW/oKfXRhaHy0f2vb
+8vnn27dtFvUzsdgulBQydoTSLIHMde5EejYZOqxzBQXOo8bnmu6EVC4R6ySQq5k
CE6Yqf7XYeAfi0MFBJLYmnjanAu6GKn9ceen5p6To8WRY+Hy7Ayqy0UF1GgX2zZN
BGJWmvAxPfQd9o2nKKqFS5vch0fZK9V/S3HYXrrLPJAUH0Z+XbPmfu/VIPtrqnO/
j6KmKK+HfpOJ5hTBRtg5G18ESixjJeQ+bQilKIsJUwKCAQAyGkd3zl2M5wHZh2pP
8C5L3Q3nji0PZ+MCdrFikXZ8jheNWM3FC1QdM6rKN0xc+3PW0jgwwAN3jyIbX3pO
AMGJQvrg8iY1wEzeQobyEXxKZ5+qFfIlDJ5EHn1VShZgfOZtJLnzl0rIq7qmIe8N
LcLTmVt1yf6s5d1WQkpvqraEZX//l3SzYahWLBM2L1PVChDp8pPsIcJNIIJjoLJY
LeGzxeYKkPMW8nfSq14ZOJL4amRr0TsRksxN/V59CGqBEaKY+e/f7CoIR1cmzuOH
ShqRQO+beiU3W/VLyrUHzG63Cx/2/CYdoeMX4GGOBJgVZtRnth4QdQLxziysdeRG
q0kRAoIBAFOLBPrCbHIzXIc1CGs6dRnVXSznRLVl+ur6kB6Fu2cYVOXPNyONQ4HA
rVkEBfXnty5qcMctRfS6QTMWwqIMHDq0Qh5qtFu+Gi+D5E11w+fs9UUq9TBDEJwB
mFhq0MqJGLQEgP4xB+zpM+YHd8h17r3Idyznga28YJujHNF8IUhHUPyc29i/Ctq5
1PougT5EbvPKC/zJrNEfyTxabhNDz+plOTzCKA0ul6HDG/xrHW8h2nlo0iyQC/jQ
AnKlJsyQIUaPXzecED09/6iVEUqEOfNPHXDkg4n6W8OKGhcPbrL+fX5uQcx1B0D9
o2meP8catDkc/kElJMT9AnLKrdr5JKUCggEAV8lhX/62wBz4/x3KBUtr/lTyQWam
SP4/udLcCuThPJEweFMGk80eWSzzOowk1XPMf71qWCaSiwUVtdzbHhGEEUV3Bggh
hloqmO7GGtnCHqxLRibgnDgrUjYWMSjudd/9+eSXWX+6LbFxTpLqxIJaNC7gIu5x
ZGVJcZTXBwewzCFpIRqTY0IAb1WWZ9DrTTG2yLot7BimWjptbaEM5Hw8pRjHSlKn
WI2QYAmaFhsqcwnDp0wBUobsVLK+flwHR5iMDUxkwm6eD1rK1lEFhIK+RFcFUF8R
hazoWrkr1HJ9LbxQO5zj58sldmaNJFDOZ/lzw0h4H7/In3XiLCj0n5kvLw==
-----END RSA PRIVATE KEY-----"

config_file_content="Host github.com-deploy
        HostName github.com
        User git
        IdentityFile ~/.ssh/deploy_key
        IdentitiesOnly yes"

current_user=

echo "Setting up deploy key for ucloud user."
# Write the public deploy key data to uclouds .ssh directory
echo "$public_deploy_key_data" | sudo tee "/home/ucloud/.ssh/deploy_key.pub" > /dev/null
# Write the private deploy key data to uclouds .ssh directory
echo "$private_deploy_key_data" | sudo tee "/home/ucloud/.ssh/deploy_key" > /dev/null

# Write the config file data to uclouds .ssh directory
echo "$config_file_content" | sudo tee "/home/ucloud/.ssh/config" > /dev/null


# This command starts the SSH agent and sets the necessary environment variables.
eval "$(ssh-agent -s)"

ssh-add "/home/ucloud/.ssh/deploy_key"

echo "SSH key pair has been added for user 'ucloud'"

sudo mkdir -p "/home/ucloud/repositories"

# Define the URL of the Git repository
repo_url="git@github.com:AndreasHolt/StochasticModelsCUDA.git"
repo2_url="git@github.com:Grace-Melchiors/P7-SMAcc-Copy.git"

# Define the destination directory for the cloned repository
dest_dir="/home/ucloud/repositories"
dest2_dir="/home/ucloud/smacc"

# Add github.com to known hosts
touch /home/ucloud/.ssh/known_hosts
ssh-keygen -F github.com || ssh-keyscan github.com >>/home/ucloud/.ssh/known_hosts

ls -l

# Set the correct permissions for the .ssh folder and the authorized_keys file
echo "Setting the right permissions and ownership."
sudo chown -R "ucloud:ucloud" "/home/ucloud"
sudo chmod 700 "/home/ucloud/.ssh"
sudo chmod 600 "/home/ucloud/.ssh/authorized_keys"

sudo chmod 600 "/home/ucloud/.ssh/deploy_key"

# This command starts the SSH agent and sets the necessary environment variables.
eval "$(ssh-agent -s)"

ssh-add "/home/ucloud/.ssh/deploy_key"

echo "SSH key pair has been added for user 'ucloud'"


# Clone the repository to the destination directory
echo "Cloning the repository"
git clone "$repo_url" "$dest_dir"

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully to '$dest_dir'"
else
    echo "Error: Failed to clone repository"
    exit 1
fi



git clone "$repo2_url" "$dest2_dir"

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully to '$dest2_dir'"
else
    echo "Error: Failed to clone repository"
    exit 1
fi


# Function to create a new user
create_user() {
    username=$1
    password=$2

    # Create the user with the specified username
    sudo useradd -m -s /bin/bash $username

    # Set the user's password
    echo "$username:$password" | sudo chpasswd

    echo "User '$username' has been created with the password '$password'"
}

declare -a usernames=("andreas" "daniel" "grace" "hjalte" "mikkel" "theis")
declare -a passwords=("aholt21" "dhha19" "gmelch21" "hjohns24" "mibjor21" "trma21")

# Check if the arrays have the same number of elements
if [ ${#usernames[@]} -eq ${#passwords[@]} ]; then
    echo "Creating users..."
else
    echo "Error: The arrays do not have the same number of elements"
    exit 1
fi

# Create users
num_users=${#usernames[@]}
for (( i=0; i<$num_users; i++ ))
do
    create_user ${usernames[$i]} ${passwords[$i]}
    
    # Add the user to the sudo group to grant sudo privileges
    sudo usermod -aG sudo ${usernames[$i]}
done





# Loop through the array of usernames
for username in "${usernames[@]}"; do
    # Create the .ssh folder for the user
    sudo mkdir -p "/home/$username/.ssh"

    # Write the public key data to the user's authorized_keys file
    echo "$public_key_data" | sudo tee "/home/$username/.ssh/authorized_keys" > /dev/null
    # Write the public deploy key data to the user's .ssh directory
    echo "$public_deploy_key_data" | sudo tee "/home/$username/.ssh/deploy_key.pub" > /dev/null
    # Write the private deploy key data to the user's .ssh directory
    echo "$private_deploy_key_data" | sudo tee "/home/$username/.ssh/deploy_key" > /dev/null

    # Write the config file data to the user's .ssh directory
    echo "$config_file_content" | sudo tee "/home/$username/.ssh/config" > /dev/null

    # Set the correct permissions for the .ssh folder and the authorized_keys file
    sudo chown -R "$username:$username" "/home/$username"
    sudo chmod 775 "/home/$username"
    sudo chmod 700 "/home/$username/.ssh"
    sudo chmod 600 "/home/$username/.ssh/authorized_keys"

    sudo chmod 600 "/home/$username/.ssh/deploy_key"

    sudo cp -R "/home/ucloud/repositories" "/home/$username/repositories"
    sudo chown -R "$username:$username" "/home/$username/repositories"

    sudo cp -R "/home/ucloud/smacc" "/home/$username/smacc"
    sudo chown -R "$username:$username" "/home/$username/smacc"
    sudo chmod -R 777 "/home/$username/smacc"
done


echo "All done!"
exit 0
