# Fixtures

Real payloads captured from the deployed pipeline, so tests assert against the
shapes the Lambdas actually exchange rather than hand-written approximations.

Each file is the output of one step and the input to the next:

| Fixture | Produced by | Consumed by |
|---|---|---|
| `s3_input_image.json` | the Step Function's start input | `serialize_image_data` |
| `imageSerializerOutput.json` | `serialize_image_data` | `predict_image_label` |
| `predictImageLabelOutput.json` | `predict_image_label` | `filter_predictions` |
