use crate::graph::{Node, SignatureIndex, Form};

pub trait ComputationProfiler {
    fn get_forward_time(&self, node: &Node, signature: SignatureIndex) -> f64;

    fn get_backward_time(&self, node: &Node, signature: SignatureIndex) -> f64 {
        self.get_forward_time(node, signature) * 2.0
    }
}

pub trait CommunicationProfiler {
    fn get_forward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64;
    fn get_backward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64;
}

pub trait Profiler {
    fn get_computation_forward_time(&self, node: &Node, signature: SignatureIndex) -> f64;
    fn get_computation_backward_time(&self, node: &Node, signature: SignatureIndex) -> f64;
    fn get_communication_forward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64;
    fn get_communication_backward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64;
}

impl<T, S> Profiler for (T, S) where T: ComputationProfiler, S: CommunicationProfiler {
    fn get_computation_forward_time(&self, node: &Node, signature: SignatureIndex) -> f64 {
        self.0.get_forward_time(node, signature)
    }

    fn get_computation_backward_time(&self, node: &Node, signature: SignatureIndex) -> f64 {
        self.0.get_backward_time(node, signature)
    }

    fn get_communication_forward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64 {
        self.1.get_forward_time(size, old_form, new_form)
    }

    fn get_communication_backward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64 {
        self.1.get_backward_time(size, old_form, new_form)
    }
}

pub struct FlopsProfiler {
    pub device_flops: u64,
    pub n_devices: usize
}

impl ComputationProfiler for FlopsProfiler {
    fn get_forward_time(&self, node: &Node, signature_index: SignatureIndex) -> f64 {
        let full_time = (node.flops as f64) / (self.device_flops as f64);
        let signature = &node.signatures[signature_index.0];
        if signature.input_forms.iter().chain(signature.output_forms.iter()).any(|form| matches!(form, &Form::Gather(_))) {
            return full_time / (self.n_devices as f64)
        }
        full_time
    }
}

pub struct BandwidthProfiler {
    pub bandwidth: u64,
    pub n_devices: usize
    // TODO: ratio of different types of communication
}

impl CommunicationProfiler for BandwidthProfiler {
    fn get_forward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64 {
        if old_form == Form::Replicate {
            return (size as f64) / (self.bandwidth as f64) / 2.
        }
        (size as f64) / (self.bandwidth as f64)
    }

    fn get_backward_time(&self, size: u64, old_form: Form, new_form: Form) -> f64 {
        if old_form == Form::Replicate {
            return (size as f64) / (self.bandwidth as f64) / 2.
        }
        (size as f64) / (self.bandwidth as f64)
    }
}
