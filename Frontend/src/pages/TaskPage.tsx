import { TaskWorkbench } from "@/components/workbench/TaskWorkbench";
import { DeepfakePage } from "@/features/deepfake/page/DeepfakePage";
import { DiarizationPage } from "@/features/task-b/page/DiarizationPage";
import { TaskDefinition } from "@/tasks/types";

interface TaskPageProps {
  task: TaskDefinition;
}

const TaskPage = ({ task }: TaskPageProps) => {
  // Audio Deepfake Detection has its own full-page layout (features/deepfake/page).
  if (task.id === "deepfake") return <DeepfakePage key={task.id} task={task} />;
  // Speaker Diarization likewise (features/task-b/page).
  if (task.id === "task-b") return <DiarizationPage key={task.id} task={task} />;
  // key forces a full workbench remount (state reset) when switching tasks
  return <TaskWorkbench key={task.id} task={task} />;
};

export default TaskPage;
