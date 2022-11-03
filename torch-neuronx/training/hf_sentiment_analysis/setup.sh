# Configure Linux for Neuron repository updates
sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

# Update OS packages. RPM may still be holding the yum lock, be patient :)
sudo yum update -y

# Install OS headers
sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

# Install Neuron SDK for Trainium
sudo yum install aws-neuronx-dkms-2.* aws-neuronx-oci-hook-2.* aws-neuronx-runtime-lib-2.* aws-neuronx-collectives-2.* aws-neuronx-tools-2.* -y

python3 -m venv trainium-hg
source trainium-hg/bin/activate

python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
pip install torch==1.11.0
pip install "numpy<=1.20.0" "protobuf<4"
pip install torch-neuronx==1.11.0.1.*
pip install neuronx-cc==2.*

echo 'export PATH=/opt/aws/neuron/bin/:$PATH' >>~/.bash_profile