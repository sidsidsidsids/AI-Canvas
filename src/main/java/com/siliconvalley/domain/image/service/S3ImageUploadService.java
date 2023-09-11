package com.siliconvalley.domain.image.service;

import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.ObjectMetadata;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
@RequiredArgsConstructor
public class S3ImageUploadService {
    private final AmazonS3Client amazonS3Client;

    @Value("${cloud.aws.s3.bucket}")
    private String bucket;

    public String uploadFile(MultipartFile multipartFile, String directory) throws IOException{
        String fileName = multipartFile.getOriginalFilename();
        String fileKey = directory + "/" + fileName;  // 키 생성

        ObjectMetadata metadata = new ObjectMetadata();
        metadata.setContentLength(multipartFile.getSize());
        metadata.setContentType(multipartFile.getContentType());

        amazonS3Client.putObject(bucket, fileKey, multipartFile.getInputStream(), metadata);
        return amazonS3Client.getUrl(bucket, fileKey).toString();
    }

    public void deleteImage(String originalFileName){
        amazonS3Client.deleteObject(bucket, originalFileName);
    }
}
