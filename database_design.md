Users {
  id string [pk]
  username string [unique]
  password_hash string
  email string [unique]
  display_name string
  avatar_url string
  role string
  created_at datetime
  updated_at datetime
}

Table RefreshTokens {
  id string [pk]
  user_id string [ref: > Users.id] 
  token string [unique]
  expires_at datetime
  is_revoked boolean [default: false]
}

Problems {
  id string [pk]
  author_id string [ref: > Users.id] 
  name string
  statement string
  theory string
  tutorial string
  solution string
  coding_template string
  input_zip_url string
  output_zip_url string
  is_public boolean [default: false]
  request_status string [default: 'NONE']
  created_at datetime
  updated_at datetime
}


Submissions {
  id string [pk]
  user_id string [ref: > Users.id]
  problem_id string [ref: > Problems.id]
  submitted_code string
  status string
  score int
  created_at datetime
}

Roadmaps {
  id string [pk]
  user_id string [ref: > Users.id] // 
  name string
  repository_url string
  level string
  user_note string
  framework string
  status string
  created_at datetime
  updated_at datetime
}

RoadmapProblems {
  id string [pk]
  roadmap_id string [ref: > Roadmaps.id]
  problem_id string [ref: > Problems.id, null] 
  name string
  description string
  order_index int
  status string 
  created_at datetime
  updated_at datetime
}