# QBrain Relay – Available Cases

## COLLECT_INFORMATION

Collect Information - gather required data from user for a target case

**Required:** user_id, session_id, target_case, missing_keys

## ANALYZE_SIM_RESULTS

Analyze simulation results from env param time-series

**Required:** user_id, goal_id

## START_RESEARCH

Start research workflow

**Required:** user_id, session_id

## GET_ENV

Retrieve a single environment by ID

**Required:** env_id

## GET_USERS_ENVS

List all environments for a user

**Required:** user_id

## DEL_ENV

Delete an environment

**Required:** env_id, user_id

## SET_ENV

Create or update an environment

**Required:** env, user_id, original_id

## DOWNLOAD_MODEL

Download model for an environment

**Required:** env_id, user_id

## RETRIEVE_LOGS_ENV

Retrieve logs for an environment

**Required:** env_id, user_id

## GET_ENV_DATA

Get environment data

**Required:** env_id, user_id

## SET_FIELD

Set field data

**Required:** field, user_id, original_id

## GET_FIELD

Get field by ID

**Required:** field_id, user_id

## GET_USERS_FIELDS

List fields for a user

**Required:** user_id

## SPAWN_OBJECT

Spawn an object instance in a given environment

**Required:** user_id, env_id, object_id, position

## GET_AVAILABLE_OBJECTS

List spawnable objects for control engine

**Required:** user_id, env_id

## LIST_USERS_PARAMS

Get Users Params

**Required:** user_id

## SET_PARAM

Set Param

**Required:** user_id, original_id, param

## DEL_PARAM

Delete Param

**Required:** user_id, param_id

## LINK_FIELD_PARAM

Link Field Param

**Required:** user_id, links

## RM_LINK_FIELD_PARAM

Rm Link Field Param

**Required:** user_id, field_id, param_id

## GET_FIELDS_PARAMS

Get Fields Params

**Required:** user_id, field_id

## SET_MODULE

Set module data

**Required:** user_id, module

## SET_FILE

Set File (Module from File)

**Required:** user_id, original_id, id, files, name, description, prompt, msg

## SET_METHOD

Set method data

**Required:** user_id, data, original_id

## SET_SESSION

Set session

**Required:** user_id, session

## SET_INJECTION

Set injection

**Required:** user_id, injection
