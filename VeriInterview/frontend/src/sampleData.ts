export const SAMPLE_CANDIDATE_NAME = "Priya Sharma";

export const SAMPLE_JOB_DESCRIPTION =
  "Senior Distributed Systems Engineer — responsible for building scalable backend microservices, event-driven pipelines on AWS (Lambda, S3, SQS), Docker/Kubernetes container orchestration, PostgreSQL optimization, and REST API design.";

export const SAMPLE_RESUME_TEXT = `PRIYA SHARMA
Senior Software Engineer

Contact: priya.sharma@email.com | LinkedIn: linkedin.com/in/priyasharma

SUMMARY
Results-driven and passionate software engineer with 5+ years of experience in building scalable distributed systems. A dynamic, self-motivated team player who leverages cutting-edge technologies to deliver best-in-class solutions.

TECHNICAL SKILLS
- Languages: Python, JavaScript, TypeScript, Java, Go
- Frameworks: React, Django, Flask, FastAPI, Express.js, Spring Boot
- Cloud: AWS (EC2, S3, Lambda, IAM, CloudFront, SQS, DynamoDB, RDS, VPC, EKS)
- DevOps: Docker, Kubernetes, Terraform, Jenkins, GitHub Actions, CI/CD
- Databases: PostgreSQL, MongoDB, Redis, Elasticsearch, DynamoDB
- Machine Learning: TensorFlow, PyTorch, scikit-learn, MLflow, deep learning
- Tools: Git, Linux, Kafka, RabbitMQ, GraphQL, gRPC
- Concepts: Microservices, Event-Driven Architecture, System Design, Agile, Scrum

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2021 - Present
- Designed and built a scalable data processing pipeline using AWS Lambda, S3, and SQS handling 10,000+ files daily
- Led migration from monolithic architecture to 15 microservices running on Kubernetes (EKS)
- Implemented auto-scaling policies reducing infrastructure costs by 35%
- Built real-time monitoring dashboards using React and D3.js
- Deployed machine learning models for anomaly detection using TensorFlow and MLflow

Software Engineer | StartupXYZ | 2019 - 2021
- Developed REST APIs using FastAPI and Django serving 2M+ requests/day
- Optimized PostgreSQL queries from 45s to 200ms through indexing and query rewriting
- Implemented JWT-based authentication and RBAC authorization system
- Built CI/CD pipelines using GitHub Actions and Docker
- Contributed to blockchain-based supply chain tracking proof of concept

Junior Developer | WebAgency | 2018 - 2019
- Built responsive web applications using React, Redux, and TypeScript
- Experience with Node.js and Express for backend development
- Familiar with basic NLP concepts and data science workflows

EDUCATION
B.Tech Computer Science | IIT Delhi | 2018
- GPA: 8.7/10
- Relevant coursework: Data Structures, Algorithms, Machine Learning, Distributed Systems

CERTIFICATIONS
- AWS Solutions Architect Associate
- Kubernetes Application Developer (CKAD)
`;

export const SAMPLE_TRANSCRIPT_TEXT = `Interviewer: Welcome! Let's start with your Python experience. Can you explain what decorators are in Python and how you've used them?

Candidate: Sure! Decorators in Python are functions that modify the behavior of other functions. They use the @ syntax and are essentially higher-order functions. I've used them extensively for logging, authentication middleware in Flask and FastAPI, and for memoization using functools.lru_cache. For example, I built a custom rate-limiting decorator at my previous company that wrapped API endpoints.

Interviewer: Great. Your resume mentions experience with AWS, specifically EC2 and Lambda. Can you walk me through how you would design a serverless architecture for a data processing pipeline?

Candidate: Absolutely. I'd use S3 as the data lake where raw files land. That triggers a Lambda function for initial validation and transformation. For heavier processing, I'd use Step Functions to orchestrate multiple Lambda invocations. The results would go to DynamoDB or RDS depending on the query patterns. I've personally set up this kind of pipeline for processing CSV uploads — about 10,000 files per day — using S3 event notifications triggering Lambda, with SQS as a buffer for handling spikes.

Interviewer: You mentioned Docker and Kubernetes in your resume. What's the difference between a Docker container and a virtual machine?

Candidate: A Docker container shares the host OS kernel and isolates the application at the process level, while a VM runs its own full operating system on top of a hypervisor. Containers are much lighter — they start in seconds and use less memory. I've used Docker for packaging microservices and Kubernetes for orchestration in production. We had about 15 services running on EKS, and I wrote the Helm charts for deployment.

Interviewer: Your resume says you have experience with machine learning. Can you explain the bias-variance tradeoff?

Candidate: Hmm, bias-variance tradeoff... I think bias is when the model is too simple, like underfitting? And variance is when it's too complex, like overfitting? You need to find the right balance. I've mostly worked with pre-trained models though, not building from scratch. I used scikit-learn for some classification tasks but my ML experience is more on the deployment side with MLflow.

Interviewer: Tell me about a time you optimized a database query. What was the performance improvement?

Candidate: At my previous role, we had a reporting query that took 45 seconds. I analyzed the execution plan and found it was doing full table scans on a 50-million row table. I added composite indexes on the commonly filtered columns, rewrote some subqueries as JOINs, and added pagination. The query went down to 200 milliseconds. I also set up a Redis cache layer for the most frequently requested reports.

Interviewer: Your resume mentions React. Can you explain the virtual DOM and why it's important?

Candidate: The virtual DOM is an in-memory representation of the real DOM. When state changes, React creates a new virtual DOM tree, diffs it against the previous one, and only updates the actual DOM nodes that changed. This is important because direct DOM manipulation is expensive. I used React for building the admin dashboard at my previous company, with Redux for state management and React Router for navigation.

Interviewer: How would you handle authentication and authorization in a REST API?

Candidate: I'd use JWT tokens for authentication. The flow would be: user logs in with credentials, server validates and returns an access token and a refresh token. The access token goes in the Authorization header as a Bearer token. For authorization, I'd implement role-based access control with middleware that checks the user's permissions against the endpoint requirements. I've implemented this using FastAPI's dependency injection system and also with Express middleware in Node.js.

Interviewer: Your resume mentions experience with system design. How would you design a URL shortening service like bit.ly?

Candidate: I'd start with a simple architecture. A web server receives the long URL, generates a unique short code using base62 encoding of an auto-incrementing ID, stores the mapping in a database, and returns the short URL. For reads, it looks up the short code and redirects. For scale, I'd add Redis for caching hot URLs, a CDN for geographic distribution, and horizontal scaling of the web tier behind a load balancer. The database would need sharding eventually based on the hash of the short code.

Interviewer: Thank you, that's all for today. Any questions from your side?

Candidate: Yes, I was wondering about the team structure and what technologies you're currently using for your backend services?
`;

export function getSampleFiles(): {
  candidateName: string;
  jobDescription: string;
  resume: File;
  transcript: File;
} {
  const resume = new File([SAMPLE_RESUME_TEXT], "priya_sharma_resume.txt", {
    type: "text/plain",
  });
  const transcript = new File([SAMPLE_TRANSCRIPT_TEXT], "interview_transcript.txt", {
    type: "text/plain",
  });
  return {
    candidateName: SAMPLE_CANDIDATE_NAME,
    jobDescription: SAMPLE_JOB_DESCRIPTION,
    resume,
    transcript,
  };
}
