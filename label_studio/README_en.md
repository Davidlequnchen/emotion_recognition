# Customer Speech Emotion Labeling Guide

[中文](README.md) | English

This subproject uses Label Studio to classify the emotion in **customer audio clips that have already been segmented**.
Each clip is one labeling task, with **one submission from each of five different accounts**; there is no need to transcribe it again or draw speaker or event boundaries.

Navigation:

- **Project lead**: [1. Create a project](#1-create-a-project) → [2. Configure the template](#2-configure-the-template) → [3. Import tasks](#3-import-tasks)
- **Annotators' daily workflow**: [4. Label clips](#4-label-clips)
- **Project lead collecting results**: [5. Export results](#5-export-results)
- **Five listeners and automatic gap filling**: [6. Require five listeners per clip](#6-require-five-listeners-per-clip)
- Configuration details: [Complete XML and field explanations](template_configuration_en.md) · [Data preparation and import](data_import_en.md)

The local demo project is [SER Emotion Labeler](http://localhost:8082/projects/3/data); the home page is <http://localhost:8082>.
The screenshots below show actual use of the local Label Studio **1.23.2** instance.
The demo uses **6 public audio clips**. Selections in the screenshots demonstrate the workflow and are not human judgments of the clips' emotions.

## 1. Create a project

**Completed by the project lead.** After logging in, click **Create Project / Create new project** and enter a name, such as `SER Emotion Labeler`.
First use public demo data to check the interface and workflow. Start the production batch after the label definitions have been agreed.

![Create a project and enter its name](screenshots/01-create-project.jpg)

## 2. Configure the template

**Completed by the project lead.** The local version does not show Audio Classification in its **Audio/Speech Processing** template list,
so there is no need to keep looking for that option. Click **Custom template** in the lower-left corner, then open **Code**.
Do not choose **Speech Transcription** or **Speaker Segmentation**; those templates do not match this clip-level emotion classification task.

![The Custom template entry in the lower-left corner of the template gallery](screenshots/02-template-gallery.jpg)

1. Choose one of the protocols below and copy the **entire XML configuration** from its file.
2. Replace the existing content in the **Code** editor. Do not append it to another template.
3. Check the preview: the upper section should contain 8 emotion options—7 emotions plus Unknown / Other—and the lower section should have a separate Bad clip rejection option. Then click **Save**.
4. For an existing project, use **Settings → Labeling Interface → Code**.

| Configuration file | What annotators see |
|---|---|
| [config_audio_only.xml](config_audio_only.xml) | Audio, 8 emotion options in the upper section, and a separate Bad clip rejection option below |
| [config_with_transcript.xml](config_with_transcript.xml) | The same interface, with the existing transcript displayed as read-only text |

![The Code editor and preview of the custom emotion classification interface](screenshots/03-custom-interface.jpg)

For the complete code, explanations of each field, and task JSON mapping, see [Custom template configuration](template_configuration_en.md).
The official [Audio Classification](https://labelstud.io/templates/audio_classification.html) template is an equivalent reference for the task type;
it is not a visible button in the local template gallery.

For an initial SER labeling round, start with audio-only. If text assistance is allowed, use with-transcript throughout the batch and ask annotators to listen before referring to the text.
Do not mix the two protocols within the same batch of independent gold labels, or show model predictions.
The current 7 emotions plus `unknown` are a starting point for the pilot; the final internal label set still needs to be confirmed.
Freeze the configuration when production labeling begins, and do not casually change field names or label values.

## 3. Import tasks

**Completed by the project lead.** The demo's 6 public clips have already been imported through **Local Files → Tasks → Save & Sync**,
so you can go directly to step 4. Storage configuration is explained in [Data preparation and import](data_import_en.md).

For new batches, you can also use **Import → Upload Files** on the project's data page:

1. Upload the `tasks.json` supplied by the person preparing the data, then click **Import**.
2. Each JSON task represents one clip and includes its audio URL, existing transcript, and stable `clip_id`.
3. Before uploading, the project lead must ensure that the current service can access the audio URLs. Uploading JSON does not upload the audio files it references.
4. **Import each JSON batch once**: do not import it through both Local Files Sync and Upload Files.

![The Import dialog's Upload Files control for the prepared task JSON](screenshots/04-upload-json.jpg)

After importing, check the task count and open the first task to confirm that the audio plays.
For the with-transcript protocol, also check that the text matches the audio.
The screenshot below was taken immediately after import: 6 tasks and 0 submissions. Two workflow demonstrations were subsequently submitted to verify the emotion and rejection paths.

![The six public demo tasks immediately after import](screenshots/05-task-list.jpg)

Keep company audio, transcripts, and task JSON in an authorized company data directory, outside this repository.
If audio will not play or a transcript does not match, ask the person preparing the data to check it.
Do not put a bare local filesystem path or `file://` URL in the audio field.

## 4. Label clips

**Completed by annotators.** Log in with your own account, open the project specified by the project lead, and click **Label All Tasks** to enter the labeling queue, then:

1. Listen to the entire clip. Replay it if needed.
2. If the audio is usable, choose one option in the upper **Emotion** section. Choose **Unknown / Other** if the emotion is uncertain or outside the defined categories.
3. If the audio itself cannot be judged reliably, choose **Bad clip — skip** in the lower **Unusable audio** section. No emotion will be assigned.
4. The two paths are mutually exclusive: selecting Bad clip clears the emotion selection, and selecting an emotion clears Bad clip. Finally, click **Submit** in the lower-right corner.

For help, hover over **ⓘ** beside either section title or over an option card to read the English explanation.
You can also use Tab to focus **ⓘ**. Emotion shortcuts are `1`–`8`; Bad clip is `0`.
You still need to click Submit after selecting an option. No text entry is required.

The selection below demonstrates the workflow; **it is not a ground-truth emotion label for this clip**. The interface uses English text.

![Select an emotion or Bad clip, then click Submit in the lower-right corner](screenshots/06-label-audio.jpg)

| Emotion value | Meaning |
|---|---|
| `angry` | Angry |
| `disgusted` | Disgusted |
| `fearful` | Fearful |
| `happy` | Happy |
| `neutral` | Neutral: there are sufficient vocal cues to support a neutral judgment |
| `sad` | Sad |
| `surprised` | Surprised |
| `unknown` | The audio is usable, but the emotion is uncertain or outside the current categories |

**Bad clip is a separate rejection decision, not a ninth emotion.** Use the lower Bad clip option when severe noise, silence, overlapping speakers,
or a similar issue means the audio cannot be judged reliably. `unknown` means an uncertain emotion in usable audio;
it does not replace rejection of unusable audio, and it is not the same as `neutral`.

After selecting **Bad clip — skip**, you must still click **Submit** to save the rejection decision.
Label Studio's native **Skip** action skips the current task and is not a substitute for this rejection option.

<details>
<summary>View the Bad clip selection state and English help tooltip</summary>

After Bad clip is selected below, none of the 8 emotion options is selected.
This also demonstrates the workflow; it is not a judgment of the sample's audio quality.

![Selecting Bad clip clears every emotion option](screenshots/09-bad-clip.jpg)

Hover over, or use the keyboard to focus, **ⓘ** beside a section title to show its English explanation.

![The Emotion section's English help tooltip](screenshots/10-section-help.jpg)

</details>

## 5. Export results

**Completed by the project lead.** On the project's data page, click **Export** and choose the full **JSON** format.

![Select JSON as the export format](screenshots/07-export-json.jpg)

Then scroll to the bottom of the dialog and click **Export** to download the file. Selecting a format alone does not download it.

![Scroll to the bottom and click Export](screenshots/08-export-button.jpg)

The default export includes annotated tasks; unsubmitted tasks are not included by default.
Workflow demonstration results are only for checking the process and must not be used as training or evaluation labels.
The native JSON stores the selection as `decision`: either one of the 8 emotion values or the `bad_clip` rejection decision.
After the project lead runs the normalization script, Bad clip becomes `status="bad_clip"` and `emotion=null`, rather than an emotion label.
For the conversion command and fields, see [Data import and export](data_import_en.md#4-read-and-normalize-exports).

Keep the complete original JSON, including `data.clip_id` and each annotation's `completed_by`, `lead_time`, `result`, and `was_cancelled`.
Keep different annotators' votes for the same clip separately. Examine agreement and disagreements before deciding on training labels;
do not overwrite them with a single label immediately. Handle cancelled annotations separately.
Community Edition's UI export does not respect the current tab filters, so do not assume that it exports only the currently filtered view.

## 6. Require five listeners per clip

**Completed by the project lead for every new project.** The five-listener requirement is a project setting, separate from the audio interface XML.
The backend in the pinned local **Community 1.23.2** version supports the following settings, including updates to existing tasks.
Both XML interfaces can be used as they are. The configuration is in [project_settings.json](project_settings.json):

```json
{
  "maximum_annotations": 5,
  "overlap_cohort_percentage": 100,
  "show_overlap_first": true,
  "enable_empty_annotation": false
}
```

`5` requires five distinct annotators per clip; `100` applies this requirement to all tasks; the other settings prioritize overlap tasks and disable empty UI submissions.
Community may not expose these controls in its settings UI. Use the project lead's API credentials with the
[Update project API](https://api.labelstud.io/api-reference/api-reference/projects/update).
First exchange a Personal Access Token for a short-lived access token following the [official token instructions](https://labelstud.io/guide/access_tokens).
Legacy token authentication is disabled in this local instance, so using an old token directly returns 401. Run from the repository root:

```bash
# Set the target server, project ID, and the short-lived access token just obtained.
export LABEL_STUDIO_URL="http://localhost:8082"
export LABEL_STUDIO_PROJECT_ID="3"
export LABEL_STUDIO_AUTH_HEADER="Bearer <your-access-token>"
curl --fail-with-body --request PATCH \
  "$LABEL_STUDIO_URL/api/projects/$LABEL_STUDIO_PROJECT_ID/" \
  --header "Authorization: $LABEL_STUDIO_AUTH_HEADER" \
  --header "Content-Type: application/json" \
  --data-binary @label_studio/project_settings.json
```

Under **Organization → Add Members**, obtain the registration link and have at least five real listeners create separate accounts.
Do not share accounts. In the normal **Label All Tasks** queue:

- A clip submitted by listeners A, B, and C remains incomplete; D and E can receive it if they have not labeled it.
- An account that has submitted a clip will not receive it again in the normal queue. After five distinct accounts submit, the clip leaves the pending queue.
- With the default skip policy, drafts and native Skip / cancelled annotations do not count as completed decisions. A submitted Unknown or Bad clip counts as one listener's decision.
- Submit only one annotation per person per clip. Do not use Quick View's `+` to add duplicates. Correct your answer by editing your own original annotation, without changing another person's results.

This is a completion threshold for the normal queue, **not a hard cap on every API or Quick View operation**.
Manually added duplicate annotations do not increase the distinct-listener count and can consume queue capacity.
The lead should check for and remove accidental duplicates before continuing.
For export acceptance, count **distinct `completed_by` values** among valid, non-cancelled decisions for each clip, rather than counting annotation rows.
Public demonstration submissions are not production listener labels; use a clean project for production data.

### Can listeners see each other's results?

Keep the current defaults: the normal **Label All Tasks** queue does not load other people's answers when delivering a task,
which helps listeners judge independently. Task details / Quick View still allow access to each other's results, without adding isolation mechanisms.
Listeners should first listen and submit their own decision, then view or discuss other results.
Community cannot enforce this convention or prevent changes to another person's results; `show_annotation_history` is not a blind-labeling permission switch.
If strict blind labeling becomes necessary, use an edition with the relevant permissions or isolated deployments.

The official [feature comparison](https://labelstud.io/guide/enterprise_features) still lists overlap settings as an Enterprise feature.
The Community behavior above was verified against the local **1.23.2** code and an isolated database workflow, including
[distinct annotator counting](https://github.com/HumanSignal/label-studio/blob/1.23.2/label_studio/tasks/mixins.py)
and [task selection](https://github.com/HumanSignal/label-studio/blob/1.23.2/label_studio/projects/functions/next_task.py).
After an upgrade or deployment change, verify that three listeners leave a clip incomplete, the fifth completes it, and an account does not receive its own submitted clip again before starting production work.

## Confirm the deployment before adding a team

A local demo and a company deployment for multiple annotators serve different purposes.
`localhost` refers to the current computer; sending that address to remote annotators will not give them access to your instance.
A company team needs an accessible company server and persistent storage.

Community Edition supports multiple user accounts, but all users can see all projects and have the same functionality.
It does not provide actual task assignment or prevent users from changing another person's annotations.
Dividing work through tabs and filters is a working convention, not permission isolation or a way to enforce blind annotation.
Five-listener gap filling uses the verified normal queue described above. Strict roles, manual task assignment, review, and enforced blind labeling still require an appropriate edition or an isolated deployment;
do not assume Enterprise instructions apply to Community Edition.

## Official references

- [Labeling interface configuration and Code](https://labelstud.io/guide/setup.html)
- [JSON export](https://labelstud.io/guide/export.html)
- [Community user permissions](https://labelstud.io/guide/signup) and [Data Manager assignment limitations](https://labelstud.io/guide/manage_data.html)
- [Official support: Community Edition does not prevent annotation overwrites](https://community.labelstud.io/t/preventing-annotation-overwrite-in-label-studio-community-edition/217)
