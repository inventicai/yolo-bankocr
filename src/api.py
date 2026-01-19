from fastapi.responses import JSONResponse
import yaml
from .inference.runner import process_image_on_the_fly,process_pdf_on_the_fly
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional

app = FastAPI()

# Load config
with open(os.path.join(os.getcwd(),'configs',"default.yaml"), "r") as f:
    config = yaml.safe_load(f)

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are allowed"}
        )
    
    try:
        results = await process_pdf_on_the_fly(
            pdf_file=await file.read(),
            cfg=config,
            pdf_name=file.filename
        )
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    
    if file.content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"error": f"Only image files are allowed: {allowed_types}"}
        )
    
    try:
        result = await process_image_on_the_fly(
            image_file=await file.read(),
            cfg=config,
            image_name=file.filename
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    
# Batch processing endpoint (for multiple files)
@app.post("/process-batch")
async def process_batch(
    files: list[UploadFile] = File(...),
    file_type: Optional[str] = "auto"
):
    """
    Process multiple files in batch.
    
    - **files**: List of files to process
    - **file_type**: "auto", "pdf", or "image" (auto detects based on content type)
    """
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Limit batch size (optional)
    max_batch_size = 10
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {max_batch_size}"
        )
    
    results = []
    errors = []
    
    for file in files:
        try:
            if file_type == "pdf" or (file_type == "auto" and file.content_type == "application/pdf"):
                pdf_bytes = await file.read()
                result = await process_pdf_on_the_fly(
                    pdf_file=pdf_bytes,
                    cfg=config,
                    pdf_name=file.filename
                )
                results.append({
                    "filename": file.filename,
                    "type": "pdf",
                    "data": result,
                    "success": True
                })
            else:
                # Process as image
                image_bytes = await file.read()
                result = await process_image_on_the_fly(
                    image_file=image_bytes,
                    cfg=config,
                    image_name=file.filename
                )
                results.append({
                    "filename": file.filename,
                    "type": "image",
                    "data": result,
                    "success": True
                })
                
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return JSONResponse(
        content={
            "success": True,
            "processed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None
        },
        status_code=200
    )    