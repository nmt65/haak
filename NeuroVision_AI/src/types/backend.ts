/* Shapes of the messages sent by the Python backend (neurovision_benchmark/server). */

export interface ModelOutput {
  probability: number;
  is_present: boolean;
  latency_ms: number;
}

export type Agreement =
  | "no_data"
  | "radio_only"
  | "camera_only"
  | "agree_present"
  | "agree_absent"
  | "camera_person_radio_absent"
  | "radio_present_camera_none";

export interface WorldState {
  type: "world_state";
  timestamp: number;
  objects: { label: string; confidence: number; sector: "left" | "center" | "right"; relative_depth: number | null }[];
  unknown_sectors: ("left" | "center" | "right")[];
  camera: { is_fresh: boolean; age_s: number | null; person_count: number | null; source: "camera" | "demo" | null; detector: string };
  radio: null | {
    mode: "simulated" | "recorded" | "hardware";
    is_simulated: boolean;
    description: string;
    sample_id: string | null;
    ground_truth_presence: number | null;
    motion_energy: number | null;
    motion_detected: boolean | null;
    plot: number[][]; // 12 feature channels × 64 time steps
    position_information: string;
  };
  radio_error: string | null;
  models: { csi: ModelOutput | null; camera: ModelOutput | null; fusion: ModelOutput | null };
  fusion: { is_synchronised: boolean; agreement: Agreement };
  runtime: { run_dir: string; dataset: string; trained_on_synthetic_data: boolean; available_models: string[] };
}

export interface BackendStatus {
  runtime: WorldState["runtime"];
  csi_mode: string;
  csi_error: string | null;
  recorded_data: string | null;
  qwen_status: string;
  qwen_error: string | null;
  yolo_status: string;
  yolo_error: string | null;
  yolo_weights_present: boolean;
  yolo_latency_ms: number | null;
  camera_connected: boolean;
}
