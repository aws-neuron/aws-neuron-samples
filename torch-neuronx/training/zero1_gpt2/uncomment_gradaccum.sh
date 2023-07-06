#! /bin/bash
script_output=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
script_output+="/accelerate/accelerator.py"

experitment_grad_accum() {
	echo 'uncommenting the assersiont to run grad_accum steps > 1'
	# look for "Gradient accumulation on TPU is not supported. Pass in `gradient_accumulation_steps=1`"
	ln=$(grep -wn "NotImplementedError" $script_output | cut -d: -f1)
	let start=$ln-2
	let end=$ln+3
	let tagln=$start-1
	sed -i "${tagln}a        \\ #ExperimentalHackOn" $script_output
	while [[ start -le $end ]]
	do
    		sed -i "$start s/./#&/" $script_output
    		((start = start + 1))
	done
}

if grep -r 'ExperimentalHackOn' $script_output; then
	echo Already edited the accelerator code
else
	echo Editing accelerator code
	experitment_grad_accum
fi
