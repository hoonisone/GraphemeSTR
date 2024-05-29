### DB
각 DB는 이름에 맞는 폴더를 하나 지정 받고 모든 데이터를 저장한다.
모든 경로는 이식성을 위해 자신이 속한 폴더에 상대 경로로 표현된다.
단 경로를 유연하게 사용할 수 있도록 relative_to 기능을 지원

project_path: db가 사용되는 프로젝트 경로
dir_path: 각 디비에 배정된 디렉터리 이름 (project_path의 상대 경로)

relative_to = "project" => 경로 데이터가 있는 경우 project_path의 상대 경로로 반환 
relative_to = "dir" => 경로 데이터가 있는 경우 dir_path의 상대 경로로 반환

get_all_id

### DATASET DB
데이터 셋 (데이터, 레이블)를 관리

### LABELSET DB
labelset이란 dataset을 합치고 train, val, test로 분할된 데이터 셋이며 실제 데이터는 없고 dataset의 경로를 이용하며, 레이블만 따로 갖는다.

### MODEL DB
딥러닝 모델을 관리하는 DB

### WORK DB
모델 + 데이터 + 학습 config
train, eval, test 등 기능 추상화

