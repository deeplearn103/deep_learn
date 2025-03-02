#Stage 1: Build

FROM python:3.9-slim as builder

#install system dependencies

RUN apt-get update &&\
    apt-get install -y --no-install-recommends gcc python3-dev &&\
    rm -rf /var/lib/apt/lists/

#install python dependencies
COPY req_prod.txt .
RUN pip install --user -r req_prod.txt

#stage2: Final
FROM python:3.9-slim

WORKDIR /app

#copy only necessary files from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

#set Environment variables
ENV PATH=/root/.local/bin:$PATH
ENV STREAMLIT_SERVER_PORT=8501

#expose port
EXPOSE 8501

#Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501",'--server.address=0.0.0.0']
